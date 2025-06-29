// dpdk_l2fwd_ml_example.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/queue.h>
#include <setjmp.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_string_fns.h>

#include "ml_onnx_async.h"

#define RTE_LOGTYPE_L2FWD RTE_LOGTYPE_USER1

#define MAX_PKT_BURST 32
#define MEMPOOL_CACHE_SIZE 256
#define FEATURES 25
#define CLASSES 2

/* Per-lcore statistics */
struct lcore_stats {
    uint64_t rx_pkts;
    uint64_t tx_pkts;
    uint64_t dropped_pkts;
    uint64_t ml_requests;
    uint64_t ml_completed;
} __rte_cache_aligned;

/* Application configuration */
struct app_config {
    unsigned n_ports;
    unsigned portmask;
    unsigned rx_queue_per_lcore;
    unsigned timer_period;
    ml_framework_t* ml_framework;
    ml_model_t* traffic_model;
} __rte_cache_aligned;

static volatile bool force_quit;
static struct lcore_stats lcore_stats[RTE_MAX_LCORE];
static struct app_config app_conf;

/* Extract packet features for ML inference */
static void extract_packet_features(struct rte_mbuf *m, float *features) {
    struct rte_ether_hdr *eth_hdr;
    struct rte_ipv4_hdr *ip_hdr;
    struct rte_tcp_hdr *tcp_hdr;
    struct rte_udp_hdr *udp_hdr;
    
    eth_hdr = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
    
    // Initialize features with default values
    memset(features, 0, FEATURES * sizeof(float));
    
    // Basic packet features
    features[0] = (float)rte_pktmbuf_pkt_len(m);
    features[1] = (float)m->nb_segs;
    
    // Check if IPv4
    if (eth_hdr->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
        ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
        
        // IP features
        features[2] = (float)ip_hdr->src_addr;
        features[3] = (float)ip_hdr->dst_addr;
        features[4] = (float)ip_hdr->next_proto_id;
        features[5] = (float)rte_be_to_cpu_16(ip_hdr->total_length);
        features[6] = (float)ip_hdr->time_to_live;
        
        // Check for TCP/UDP
        if (ip_hdr->next_proto_id == IPPROTO_TCP) {
            tcp_hdr = (struct rte_tcp_hdr *)((unsigned char *)ip_hdr + 
                      ((ip_hdr->version_ihl & 0x0f) * 4));
            features[7] = (float)rte_be_to_cpu_16(tcp_hdr->src_port);
            features[8] = (float)rte_be_to_cpu_16(tcp_hdr->dst_port);
            features[9] = (float)tcp_hdr->tcp_flags;
            features[10] = (float)rte_be_to_cpu_32(tcp_hdr->sent_seq);
        } else if (ip_hdr->next_proto_id == IPPROTO_UDP) {
            udp_hdr = (struct rte_udp_hdr *)((unsigned char *)ip_hdr + 
                      ((ip_hdr->version_ihl & 0x0f) * 4));
            features[7] = (float)rte_be_to_cpu_16(udp_hdr->src_port);
            features[8] = (float)rte_be_to_cpu_16(udp_hdr->dst_port);
            features[11] = (float)rte_be_to_cpu_16(udp_hdr->dgram_len);
        }
    }
    
    // Timing features
    features[12] = (float)rte_rdtsc();
    
    // Additional features can be extracted as needed
}

/* ML inference callback */
struct inference_context {
    struct rte_mbuf *mbuf;
    uint16_t port_id;
    unsigned lcore_id;
};

static void packet_inference_callback(void* user_data, float* output, 
                                    size_t output_size, int status) {
    struct inference_context *ctx = (struct inference_context *)user_data;
    
    if (status == ML_STATUS_SUCCESS) {
        // Determine traffic class based on ML output
        int traffic_class = (output[0] > output[1]) ? 0 : 1;
        
        // Apply QoS based on classification
        if (traffic_class == 0) {
            // High priority traffic
            ctx->mbuf->hash.rss = 0;  // Can be used for QoS marking
        } else {
            // Low priority traffic
            ctx->mbuf->hash.rss = 1;
        }
        
        lcore_stats[ctx->lcore_id].ml_completed++;
    }
    
    // Mark mbuf as ready for transmission
    ctx->mbuf->ol_flags |= PKT_TX_IPV4;
}

/* L2 forwarding with ML inference */
static void l2fwd_ml_forward(struct rte_mbuf *m, unsigned dst_port, 
                            unsigned lcore_id) {
    struct rte_ether_hdr *eth;
    void *tmp;
    float features[FEATURES];
    
    eth = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
    
    /* Perform MAC swap for L2 forwarding */
    tmp = &eth->d_addr.addr_bytes[0];
    *((uint64_t *)tmp) = 0x000000000002 + ((uint64_t)dst_port << 40);
    
    /* Extract features and submit ML inference */
    extract_packet_features(m, features);
    
    /* Allocate context for callback */
    struct inference_context *ctx = rte_malloc("inf_ctx", 
                                              sizeof(struct inference_context), 
                                              0);
    if (ctx) {
        ctx->mbuf = m;
        ctx->port_id = dst_port;
        ctx->lcore_id = lcore_id;
        
        /* Submit async inference with high priority and 5ms timeout */
        uint64_t req_id = ml_inference_async(
            app_conf.ml_framework,
            app_conf.traffic_model,
            features,
            sizeof(features),
            packet_inference_callback,
            ctx,
            ML_PRIORITY_HIGH,
            5  /* 5ms timeout for low latency */
        );
        
        if (req_id > 0) {
            lcore_stats[lcore_id].ml_requests++;
        } else {
            /* Failed to submit, forward without classification */
            rte_free(ctx);
        }
    }
}

/* Main packet processing loop */
static int l2fwd_ml_launch_one_lcore(void *arg) {
    struct app_config *conf = (struct app_config *)arg;
    unsigned lcore_id = rte_lcore_id();
    struct rte_mbuf *pkts_burst[MAX_PKT_BURST];
    struct rte_mbuf *m;
    unsigned i, j, nb_rx, nb_tx;
    uint16_t port_id;
    
    RTE_LOG(INFO, L2FWD, "Entering main loop on lcore %u\n", lcore_id);
    
    /* Register this lcore as PP core */
    ml_register_pp_core(conf->ml_framework, lcore_id);
    
    /* Main processing loop */
    while (!force_quit) {
        /* RX processing */
        for (port_id = 0; port_id < conf->n_ports; port_id++) {
            if ((conf->portmask & (1 << port_id)) == 0)
                continue;
            
            nb_rx = rte_eth_rx_burst(port_id, 0, pkts_burst, MAX_PKT_BURST);
            lcore_stats[lcore_id].rx_pkts += nb_rx;
            
            /* Prefetch packets */
            for (j = 0; j < nb_rx; j++) {
                rte_prefetch0(rte_pktmbuf_mtod(pkts_burst[j], void *));
            }
            
            /* Process and forward packets with ML inference */
            for (j = 0; j < nb_rx; j++) {
                m = pkts_burst[j];
                l2fwd_ml_forward(m, port_id ^ 1, lcore_id);
            }
        }
        
        /* Poll for ML inference responses */
        ml_async_request_t* responses[MAX_PKT_BURST];
        int nb_responses = ml_poll_responses_bulk(conf->ml_framework, lcore_id,
                                                 responses, MAX_PKT_BURST);
        
        /* Process completed inferences and TX */
        for (i = 0; i < nb_responses; i++) {
            struct inference_context *ctx = 
                (struct inference_context *)responses[i]->user_data;
            
            /* Send packet */
            nb_tx = rte_eth_tx_burst(ctx->port_id, 0, &ctx->mbuf, 1);
            if (nb_tx > 0) {
                lcore_stats[lcore_id].tx_pkts += nb_tx;
            } else {
                lcore_stats[lcore_id].dropped_pkts++;
                rte_pktmbuf_free(ctx->mbuf);
            }
            
            /* Cleanup */
            ml_free_input_buffer(conf->ml_framework, responses[i]->input_buffer);
            ml_free_output_buffer(conf->ml_framework, responses[i]->output_buffer);
            rte_free(ctx);
            free_request(responses[i]);
        }
    }
    
    return 0;
}

/* Display statistics */
static void print_stats(void) {
    uint64_t total_rx = 0, total_tx = 0, total_dropped = 0;
    uint64_t total_ml_req = 0, total_ml_comp = 0;
    unsigned lcore_id;
    
    printf("\n==== L2FWD-ML Statistics ====\n");
    RTE_LCORE_FOREACH(lcore_id) {
        printf("Lcore %u:\n", lcore_id);
        printf("  RX packets: %"PRIu64"\n", lcore_stats[lcore_id].rx_pkts);
        printf("  TX packets: %"PRIu64"\n", lcore_stats[lcore_id].tx_pkts);
        printf("  Dropped: %"PRIu64"\n", lcore_stats[lcore_id].dropped_pkts);
        printf("  ML requests: %"PRIu64"\n", lcore_stats[lcore_id].ml_requests);
        printf("  ML completed: %"PRIu64"\n", lcore_stats[lcore_id].ml_completed);
        
        total_rx += lcore_stats[lcore_id].rx_pkts;
        total_tx += lcore_stats[lcore_id].tx_pkts;
        total_dropped += lcore_stats[lcore_id].dropped_pkts;
        total_ml_req += lcore_stats[lcore_id].ml_requests;
        total_ml_comp += lcore_stats[lcore_id].ml_completed;
    }
    
    printf("\nTotal:\n");
    printf("  RX: %"PRIu64"\n", total_rx);
    printf("  TX: %"PRIu64"\n", total_tx);
    printf("  Dropped: %"PRIu64"\n", total_dropped);
    printf("  ML requests: %"PRIu64"\n", total_ml_req);
    printf("  ML completed: %"PRIu64"\n", total_ml_comp);
    
    /* ML Framework stats */
    char ml_stats[4096];
    ml_get_stats(app_conf.ml_framework, ml_stats, sizeof(ml_stats));
    printf("\n%s\n", ml_stats);
}

/* Signal handler */
static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        printf("\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}

/* Initialize ML framework and model */
static int init_ml_framework(void) {
    /* ML Framework configuration */
    ml_framework_config_t ml_config = {
        .num_pp_cores = rte_lcore_count() - 2,  /* Reserve 2 cores for ML */
        .memory = {
            .input_buffer_size = 1024,
            .output_buffer_size = 256,
            .num_input_buffers = 4096,
            .num_output_buffers = 4096,
            .use_hugepages = true
        },
        .default_timeout_ms = 10,
        .enable_profiling = true,
        .max_concurrent_requests = 2048
    };
    
    /* Initialize ML framework */
    app_conf.ml_framework = ml_framework_init(&ml_config);
    if (!app_conf.ml_framework) {
        RTE_LOG(ERR, L2FWD, "Failed to initialize ML framework\n");
        return -1;
    }
    
    /* Define model configuration */
    tensor_info_t input_tensor = {
        .name = "X",
        .dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        .shape = {1, FEATURES},
        .num_dimensions = 2,
        .total_elements = FEATURES,
        .is_dynamic = false
    };
    
    tensor_info_t output_tensor = {
        .name = "probabilities",
        .dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        .shape = {1, CLASSES},
        .num_dimensions = 2,
        .total_elements = CLASSES,
        .is_dynamic = false
    };
    
    ml_model_config_t model_config = {
        .model_path = "rf_10_pkts_model_sklearn.onnx",
        .model_name = "traffic_classifier",
        .model_type = MODEL_TYPE_RANDOM_FOREST,
        .description = "Random Forest traffic classifier",
        .input_tensors = &input_tensor,
        .num_inputs = 1,
        .output_tensors = &output_tensor,
        .num_outputs = 1,
        .max_batch_size = 32
    };
    
    /* Load model */
    app_conf.traffic_model = ml_load_model(app_conf.ml_framework, &model_config);
    if (!app_conf.traffic_model) {
        RTE_LOG(ERR, L2FWD, "Failed to load ML model\n");
        return -1;
    }
    
    RTE_LOG(INFO, L2FWD, "ML framework initialized successfully\n");
    ml_print_model_info(app_conf.traffic_model);
    
    return 0;
}

int main(int argc, char **argv) {
    int ret;
    uint16_t nb_ports;
    unsigned lcore_id;
    
    /* Initialize EAL */
    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Invalid EAL arguments\n");
    
    argc -= ret;
    argv += ret;
    
    force_quit = false;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* Initialize ML framework */
    if (init_ml_framework() < 0)
        rte_exit(EXIT_FAILURE, "ML framework initialization failed\n");
    
    /* Check available ports */
    nb_ports = rte_eth_dev_count_avail();
    if (nb_ports < 2)
        rte_exit(EXIT_FAILURE, "Requires at least 2 ports\n");
    
    app_conf.n_ports = nb_ports;
    app_conf.portmask = (1 << nb_ports) - 1;
    
    /* Launch workers on all available lcores */
    RTE_LCORE_FOREACH_SLAVE(lcore_id) {
        rte_eal_remote_launch(l2fwd_ml_launch_one_lcore, &app_conf, lcore_id);
    }
    
    /* Main lcore also processes packets */
    l2fwd_ml_launch_one_lcore(&app_conf);
    
    /* Wait for all lcores to finish */
    RTE_LCORE_FOREACH_SLAVE(lcore_id) {
        if (rte_eal_wait_lcore(lcore_id) < 0) {
            ret = -1;
