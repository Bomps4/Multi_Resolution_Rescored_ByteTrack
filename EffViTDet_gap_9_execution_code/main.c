#include "onnx_graph.h"
#include "onnx_graphKernels.h"
#include "Low_size_onnx_graphKernels.h"
#include "bsp/app_bin.h"
#include <pmsis.h>
#include "gaplib/fs_switch.h"
#include "measurments_utils.h"



#define TEST_GPIO_0 PI_GPIO_A89
#define TEST_GPIO_1 PI_GPIO_A68
#define TEST_GPIO_2 PI_GPIO_A52
#define WRITE_GPIO(gpio_pin_x, x) {hal_compiler_barrier(); pi_gpio_pin_write(gpio_pin_x, x); hal_compiler_barrier();}
#define SWITCH_GPIO(gpio_pin_x) {hal_compiler_barrier(); pi_gpio_pin_toggle(gpio_pin_x); hal_compiler_barrier();}




AT_DEFAULTFLASH_EXT_ADDR_TYPE onnx_graph_L3_Flash = 0;
AT_DEFAULTFLASH_EXT_ADDR_TYPE Low_size_onnx_graph_L3_Flash = 0;

pi_gpio_e gpio_test_0, gpio_test_1, gpio_test_2;



pi_err_t app_description_load(pi_device_e *flash_ref, app_bin_description_t *app_description)
{
    pi_fpv2_ptable_conf_t pt_conf;
    pi_fpv2_ptable_conf_init(&pt_conf);
    pi_device_e mram = PI_FLASH_MRAM;
    pi_err_t rc = pi_fpv2_ptable_offset_get(mram, &pt_conf.offset, 0);
    if (PI_OK != rc)
    {
        // printf("Failed to load the partition table (rc=%d)\n", rc);
        return -3;
    }

    /* load the partition table */
    pi_fpv2_ptable_desc_t* ptable_desc = NULL;
    rc = pi_fpv2_ptable_load(&ptable_desc, &pt_conf);
    if (PI_OK != rc)
    {
        // printf("Failed to load the partition table (rc=%d)\n", rc);
        return -4;
    }

    /* load the volume table */
    pi_fpv2_vtable_desc_t* vtable_desc = NULL;
    rc = pi_fpv2_vtable_load(ptable_desc, &vtable_desc);
    if (PI_OK != rc)
    {
        // printf("Failed to load the volume table(rc=%d)\n", rc);
        return -5;
    }

    /* load the active volume */
    pi_fpv2_volume_desc_t* app_volume_desc = NULL;
    uint8_t valid_app = 0;
    while (valid_app == 0) // FIXME put a max iteration number ?
    {
        rc = pi_fpv2_vtable_active_application_volume_get(vtable_desc, &app_volume_desc);
        //rc = pi_fpv2_vtable_volume_get_by_label(vtable_desc, "app", &app_volume_desc);
        if (PI_OK != rc)
        {
            // printf("Failed to load the active app volume (rc=%d)\n", rc);
            return -6;
        }

        /* check the boot counter */
        uint8_t boot_counter = 0x0;
        rc = pi_fpv2_volume_boot_counter_get(app_volume_desc, &boot_counter);
        if (PI_OK != rc)
        {
            // printf("Failed to get the boot counter (rc=%d)\n", rc);
            return -7;
        }

        if (boot_counter > 10) //TODO make this configurable + avoid magic value
        {
            /* App failed to boot many times in a row, it's considered invalid */
            // printf("App failed to boot many times in a row, mark it invalid\n");
            pi_fpv2_volume_boot_counter_set_invalid(app_volume_desc);
        }
        else
        {
            valid_app = 1;
        }
    }

    /* find the partition */
    pi_fpv2_partition_desc_t* part_desc = NULL;
    rc = pi_fpv2_volume_partition_first_get_by_type(app_volume_desc,
            PI_FPV2_PARTITION_TYPE_APP, PI_FPV2_PARTITION_SUBTYPE_APP_BIN,
            &part_desc);

    if (PI_OK != rc)
    {
        // printf("Failed to load the app_binary partition (rc=%d)\n", rc);
        return -8;
    }

    uint32_t app_addr;
    pi_fpv2_partition_info_t part_info;
    rc = pi_fpv2_ptable_partition_info_get(part_desc, &part_info);
    if (PI_OK == rc)
    {
        app_addr = part_info.gref.lref.offset;
    }
    else
    {
        // printf("Failed to get partition info (rc=%d)\n", rc);
        return -9;
    }

    uint32_t offset = app_addr;
    // printf("offset: 0x%08x\n", offset);
    /* get application description */
    // printf("Get App ELF description\n");
    pi_fpv2_ptable_partition_device_get(part_desc, flash_ref);
    rc = pi_app_bin_description_get(*flash_ref, app_addr, app_description);
    return rc;
}



/* Inputs */
/* Outputs */

/* Copy inputs functions */
switch_fs_t input_fs;
void *Input_File_Input_1;
int Input_File_Input_1_Position;
#define EXPECTED_NUM_ITERATIONS 2
int open_inputs() {
    // __FS_INIT(input_fs);

    // /* opening file Input_1 */
    // #ifdef __EMUL__
    // Input_File_Input_1 = __OPEN_READ(input_fs, "../Input_1.bin");
    // #else
    // Input_File_Input_1 = __OPEN_READ(input_fs, "../Input_1.bin");
    // #endif
    // if (!Input_File_Input_1) return 1;
    // Input_File_Input_1_Position = 0;
    return 0;
}

int copy_inputs(int num_iterations) {
    /* Reading from file Input_1 */
    int ret_Input_1 = 0;
    __SEEK(Input_File_Input_1, Input_File_Input_1_Position);
    ret_Input_1 = __READ(Input_File_Input_1, Input_1, 221184);
    if (ret_Input_1 != 221184) {
        return 0;
    }
    Input_File_Input_1_Position = 0;
    return 1;
}

void close_inputs() {
    __CLOSE(Input_File_Input_1);

    __FS_DEINIT(input_fs);
}

/* Copy outputs functions */
switch_fs_t output_fs;
int open_outputs() {
    return 0;
}

void write_outputs() {
}

void close_outputs() {
}





static void cluster(void * arg)
{   
    // printf("Start timer\n");
    #ifdef PERF
   
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif
    int UseLargeModel = (int) arg;

    // Low_size_onnx_graphCNN();

    // // onnx_graphCNN_ConstructCluster();
    if (UseLargeModel)
        {

        onnx_graphCNN();

    
        }
    else
        {
        
        
        S1_input_1_resizer((f16*)(Low_size_onnx_graph_L2_Memory_Dyn + 221184),Low_size_Input_1);
        
        Low_size_onnx_graphCNN();
        

        }

    // printf("Runner completed \n");
}


pi_device_e flash_ref;
app_bin_description_t app_description;

int main(int argc, char *argv[])
{
    // printf("\n\n\t *** NNTOOL onnx_graph Example ***\n\n");
    // printf("Entering main controller\n");

    gpio_test_0 = TEST_GPIO_0;
    pi_pad_function_set(PI_PAD_089, 1);
    pi_gpio_pin_configure(gpio_test_0, PI_GPIO_OUTPUT);
    WRITE_GPIO(gpio_test_0, 1);

    gpio_test_1 = TEST_GPIO_1;
    pi_pad_function_set(PI_PAD_068, 1);
    pi_gpio_pin_configure(gpio_test_1, PI_GPIO_OUTPUT);
    WRITE_GPIO(gpio_test_1, 1);

    gpio_test_2 = TEST_GPIO_2;
    pi_pad_function_set(PI_PAD_052, 1);
    pi_gpio_pin_configure(gpio_test_2, PI_GPIO_OUTPUT);
    WRITE_GPIO(gpio_test_2, 1);


    pi_err_t rc = app_description_load(&flash_ref, &app_description);
    if(rc)
    {
        // printf("Error loadind app.\n");
        pmsis_exit(rc);
    }
    // if(pi_app_bin_section_load_by_name(flash_ref, ".effvit_large", &app_description))
    // {
    //     printf("Failed to load section.\n");
    //     pmsis_exit(rc);
    // }


    /* Configure And open cluster. */
    
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;

    cl_conf.id = 0; /* Set cluster ID. */
                    // Enable the special icache for the master core
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |
                    // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                    PI_CLUSTER_ICACHE_PREFETCH_ENABLE |
                    // Enable the icache for all the cores
                    PI_CLUSTER_ICACHE_ENABLE;

    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        // printf("Cluster open failed !\n");
        return -4;
    }

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        // printf("Error changing frequency !\nTest failed...\n");
        return -4;
    }
	// printf("FC Frequency = %d Hz CL Frequency = %d Hz PERIPH Frequency = %d Hz\n", 
    //         pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

	#ifdef VOLTAGE
	pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
	pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
	#endif
	// printf("Voltage: %dmV\n", pi_pmu_voltage_get(PI_PMU_VOLTAGE_DOMAIN_CHIP));

    


    

    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = onnx_graphCNN_Construct(1);
    if (ConstructorErr)
    {
        // printf("Graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorErr));
        return -6;
    }
    

    /*
     * Put here Your input settings
     */
    // if (open_inputs()) return -7;
    // if (open_outputs()) return -8;
    // printf("Call cluster\n");

    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);

    // printf("stack size %d \n",SLAVE_STACK_SIZE);
    
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    // printf("is the stack null %d \n",task.stacks==NULL);

    
    int iteration = 0;
    while (iteration < EXPECTED_NUM_ITERATIONS) {
        // printf("ma almeno partire \n");
        if (EXPECTED_NUM_ITERATIONS > 0) printf("Iteration: %d of %d\n", iteration + 1, EXPECTED_NUM_ITERATIONS);
       


        int UseLargeModel = !(iteration % 2);
        if((UseLargeModel) )
        {   
            SWITCH_GPIO(gpio_test_0);
            // printf("ma questo che dovrebbe \n");
            uint time = pi_time_get_us();
                pi_app_bin_section_load_by_name(flash_ref, ".effvit_large", &app_description);
            printf("us  %u\n",((uint)pi_time_get_us())- time);
            // printf("loading large");
            
            
        }
        else
        {
            SWITCH_GPIO(gpio_test_0);
            Low_size_onnx_graphCNN_Construct(iteration==1);
            uint time = pi_time_get_us();
            pi_app_bin_section_load_by_name(flash_ref, ".effvit_small", &app_description);
            printf("us %u\n",((uint)pi_time_get_us()) - time);
            // printf("loading small");
            
        }   


        // if((UseLargeModel) )
        // SWITCH_GPIO(gpio_test_1)
        // else
        //     SWITCH_GPIO(gpio_test_1)

        task.arg = (void*)UseLargeModel;
        SWITCH_GPIO(gpio_test_0);
        // printf("e la conversione in void* \n");

        pi_cluster_send_task_to_cl(&cluster_dev, &task);
        
        // printf("Qua 1");
        // if((UseLargeModel) )
        //     SWITCH_GPIO(gpio_test_0)
        // else
        //     SWITCH_GPIO(gpio_test_1);
        // printf("Qua 2");
        iteration++;
    }
    // close_inputs();
    // close_outputs();
    
    
    // onnx_graphCNN_Destruct();
#ifdef PERF

	{
		unsigned long long int TotalCycles = 0, TotalOper = 0;
		// printf("\n");
		for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
		}
		for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			// printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
		}
		// printf("\n");
		// printf("%45s: Cycles: %12llu, Cyc%%: 100.0%%, Operations: %12llu, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
		// printf("\n");
	}
#endif

    printf("Ended\n");
    return 0;
}
