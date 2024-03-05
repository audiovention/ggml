#include "ggml-wgpu.h"
#include "ggml.h"

#include <webgpu/webgpu.h>
#ifdef WEBGPU_BACKEND_WGPU
#include <webgpu/wgpu.h>
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifdef GGML_PERF
#define ggml_perf_time_ms()       ggml_time_ms()
#define ggml_perf_time_us()       ggml_time_us()
#define ggml_perf_cycles()        ggml_cycles()
#define ggml_perf_cycles_per_ms() ggml_cycles_per_ms()
#else
#define ggml_perf_time_ms()       0
#define ggml_perf_time_us()       0
#define ggml_perf_cycles()        0
#define ggml_perf_cycles_per_ms() 0
#endif


#define GGML_WGPU_DST_BINDING_INDEX (GGML_MAX_SRC)
#define GGML_WGPU_DIM_PARAMS_BINDING_INDEX (GGML_WGPU_DST_BINDING_INDEX+1)
#define GGML_WGPU_NUM_EXTRA_UNIFORM_BINDINGS (2)
#define GGML_WGPU_EXTRA_UNIFORM_SIZE (4*4*1024)
#define GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX (GGML_WGPU_DIM_PARAMS_BINDING_INDEX+1)
#define GGML_WGPU_BINDINGS_SIZE (GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX+GGML_WGPU_NUM_EXTRA_UNIFORM_BINDINGS)
#define GGML_WGPU_DIM_PARAMS_SIZE (GGML_MAX_SRC+1)
#define GGML_WGPU_OP_PARAMS_SIZE (12) // should be "GGML_MAX_OP_PARAMS / sizeof(int32_t)" but we round it up to 12 to make sure it is aligned in accordance with the wgsl struct requirements..

#define MIN_STORAGE_BUFFER_ALIGNMENT 256
#define UNUSED(x) (void)(x)
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define PLACEHOLDER_BUFFER_SIZE (8*(4*1024*MIN_STORAGE_BUFFER_ALIGNMENT))

#define ASSERT_CHECK(x) \
    if (!(x)) { \
        GGML_WGPU_LOG_ERROR("%s: error: assertion failed: %s\n", __func__, #x); \
        return NULL; \
    }

#define LOG_PREFIX "[compute]"
static void handle_request_adapter(WGPURequestAdapterStatus status,
                                   WGPUAdapter adapter, char const *message,
                                   void *userdata) {
  UNUSED(status);
  UNUSED(message);
  *(WGPUAdapter *)userdata = adapter;
}
static void handle_request_device(WGPURequestDeviceStatus status,
                                  WGPUDevice device, char const *message,
                                  void *userdata) {
  UNUSED(status);
  UNUSED(message);
  *(WGPUDevice *)userdata = device;
}

#ifdef WEBGPU_BACKEND_DAWN
#include "time.h"
void sleep_ms(int milliseconds)
{
    #ifdef WIN32
        Sleep(milliseconds);
    #elif _POSIX_C_SOURCE >= 199309L
        struct timespec ts;
        ts.tv_sec = milliseconds / 1000;
        ts.tv_nsec = (milliseconds % 1000) * 1000000;
        nanosleep(&ts, NULL);
    #else
        usleep(milliseconds * 1000);
    #endif
}
#endif


#define MULTILINE(...) #__VA_ARGS__

static const char src_ggml_shader_common_0[] = MULTILINE(

enable f16;

struct TensorDimensionParam {
        ne : vec4i,
        nb : vec4u,
        @size(16) offset : u32,
}


struct TensorDimensionParams {
        src : array<TensorDimensionParam, 6>,
        dst : TensorDimensionParam,
        params : array<vec4i, 3>,
}



@group(0) @binding(0)
var<storage,read_write> src0: array<f16>;

@group(0) @binding(1)
var<storage,read_write> src1: array<f16>;

@group(0) @binding(2)
var<storage,read_write> src2: array<f16>;

@group(0) @binding(3)
var<storage,read_write> src3: array<f16>;

@group(0) @binding(4)
var<storage,read_write> src4: array<f16>;

@group(0) @binding(5)
var<storage,read_write> src5: array<f16>;

@group(0) @binding(6)
var<storage,read_write> dst: array<f16>;


@group(0) @binding(0)
var<storage,read_write> src0_v4: array<vec4h>;

@group(0) @binding(1)
var<storage,read_write> src1_v4: array<vec4h>;

@group(0) @binding(2)
var<storage,read_write> src2_v4: array<vec4h>;

@group(0) @binding(3)
var<storage,read_write> src3_v4: array<vec4h>;

@group(0) @binding(4)
var<storage,read_write> src4_v4: array<vec4h>;

@group(0) @binding(5)
var<storage,read_write> src5_v4: array<vec4h>;

@group(0) @binding(6)
var<storage,read_write> dst_v4: array<vec4h>;



@group(0) @binding(7)
var<uniform> tensor_dimension_params: TensorDimensionParams;


@group(0) @binding(8)
var<uniform> extra_uniform0: array<vec4f, 1024>;

@group(0) @binding(9)
var<uniform> extra_uniform1: array<vec4f, 1024>;


fn get_src0(x: u32, y: u32, z: u32) -> f16 {
    return src0[x 
                //   * tensor_dimension_params.src[0].nb[0]
                + y * tensor_dimension_params.src[0].nb[1] +
                z * tensor_dimension_params.src[0].nb[2]
                    // + tensor_dimension_params.src[0].offset
                    ];
}

fn get_src1(x: u32, y: u32, z: u32) -> f16 {
    return src1[x 
                //   * tensor_dimension_params.src[1].nb[0]
                + y * tensor_dimension_params.src[1].nb[1] +
                z * tensor_dimension_params.src[1].nb[2]
                    // + tensor_dimension_params.src[1].offset
                    ];
}

fn get_src2(x: u32, y: u32, z: u32) -> f16 {
    return src2[x 
                //   * tensor_dimension_params.src[2].nb[0]
                + y * tensor_dimension_params.src[2].nb[1] +
                z * tensor_dimension_params.src[2].nb[2]
                    // + tensor_dimension_params.src[2].offset
                    ];
}

fn get_src3(x: u32, y: u32, z: u32) -> f16 {
    return src3[x 
                //   * tensor_dimension_params.src[3].nb[0]
                + y * tensor_dimension_params.src[3].nb[1] +
                z * tensor_dimension_params.src[3].nb[2]
                    // + tensor_dimension_params.src[3].offset
                    ];
}

fn get_src4(x: u32, y: u32, z: u32) -> f16 {
    return src4[x 
                //   * tensor_dimension_params.src[4].nb[0]
                + y * tensor_dimension_params.src[4].nb[1] +
                z * tensor_dimension_params.src[4].nb[2]
                    // + tensor_dimension_params.src[4].offset
                    ];
}

fn get_src5(x: u32, y: u32, z: u32) -> f16 {
    return src5[x 
                //   * tensor_dimension_params.src[5].nb[0]
                + y * tensor_dimension_params.src[5].nb[1] +
                z * tensor_dimension_params.src[5].nb[2]
                    // + tensor_dimension_params.src[5].offset
                    ];
}

fn set_dst(x: u32, y: u32, z: u32, v: f16) {
    dst[ x 
        //    * tensor_dimension_params.dst.nb[0]
         + y * tensor_dimension_params.dst.nb[1] +
         z * tensor_dimension_params.dst.nb[2]
            // + tensor_dimension_params.dst.offset
             ] = v;
}

);

static const char src_ggml_shader_common_1[] = MULTILINE(


fn get_src0_lin(x: u32) -> f16 {
    return src0[x 
                //   * tensor_dimension_params.src[0].nb[0]
                    // + tensor_dimension_params.src[0].offset
                    ];
}

fn get_src1_lin(x: u32) -> f16 {
    return src1[x 
                //   * tensor_dimension_params.src[1].nb[0]
                    // + tensor_dimension_params.src[1].offset
                    ];
}

fn get_src2_lin(x: u32) -> f16 {
    return src2[x 
                //   * tensor_dimension_params.src[2].nb[0]
                    // + tensor_dimension_params.src[2].offset
                    ];
}

fn get_src3_lin(x: u32) -> f16 {
    return src3[x 
                //   * tensor_dimension_params.src[3].nb[0]
                    // + tensor_dimension_params.src[3].offset
                    ];
}

fn get_src4_lin(x: u32) -> f16 {
    return src4[x 
                //   * tensor_dimension_params.src[4].nb[0]
                    // + tensor_dimension_params.src[4].offset
                    ];
}

fn get_src5_lin(x: u32) -> f16 {
    return src5[x 
                //   * tensor_dimension_params.src[5].nb[0]
                    // + tensor_dimension_params.src[5].offset
                    ];
}

fn get_dst_lin(x: u32) -> f16 {
    return dst[ x 
        //    * tensor_dimension_params.dst.nb[0]
            // + tensor_dimension_params.dst.offset
             ];
}


fn set_dst_lin(x: u32, v: f16) {
    dst[ x 
        //    * tensor_dimension_params.dst.nb[0]
            // + tensor_dimension_params.dst.offset
             ] = v;
}

fn set_src0_lin(x: u32, v: f16) {
    src0[ x 
        //    * tensor_dimension_params.src[0].nb[0]
            // + tensor_dimension_params.src[0].offset
             ] = v;
}

fn set_src1_lin(x: u32, v: f16) {
    src1[ x 
        //    * tensor_dimension_params.src[1].nb[0]
            // + tensor_dimension_params.src[1].offset
             ] = v;
}

fn set_src2_lin(x: u32, v: f16) {
    src2[ x 
        //    * tensor_dimension_params.src[2].nb[0]
            // + tensor_dimension_params.src[2].offset
             ] = v;
}

fn set_src3_lin(x: u32, v: f16) {
    src3[ x 
        //    * tensor_dimension_params.src[3].nb[0]
            // + tensor_dimension_params.src[3].offset
             ] = v;
}

fn set_src4_lin(x: u32, v: f16) {
    src4[ x 
        //    * tensor_dimension_params.src[4].nb[0]
            // + tensor_dimension_params.src[4].offset
             ] = v;
}

fn set_src5_lin(x: u32, v: f16) {
    src5[ x 
        //    * tensor_dimension_params.src[5].nb[0]
            // + tensor_dimension_params.src[5].offset
             ] = v;
}

);

static const char src_ggml_shader_common_2[] = MULTILINE(


fn set_src0(x: u32, y: u32, z: u32, v: f16) {
    src0[ x 
        //    * tensor_dimension_params.src[0].nb[0]
         + y * tensor_dimension_params.src[0].nb[1] +
         z * tensor_dimension_params.src[0].nb[2]
            // + tensor_dimension_params.src[0].offset
             ] = v;
}

fn set_src1(x: u32, y: u32, z: u32, v: f16) {
    src1[ x 
        //    * tensor_dimension_params.src[1].nb[0]
         + y * tensor_dimension_params.src[1].nb[1] +
         z * tensor_dimension_params.src[1].nb[2]
            // + tensor_dimension_params.src[1].offset
             ] = v;
}

fn set_src2(x: u32, y: u32, z: u32, v: f16) {
    src2[ x 
        //    * tensor_dimension_params.src[2].nb[0]
         + y * tensor_dimension_params.src[2].nb[1] +
         z * tensor_dimension_params.src[2].nb[2]
            // + tensor_dimension_params.src[2].offset
             ] = v;
}

fn set_src3(x: u32, y: u32, z: u32, v: f16) {
    src3[ x 
        //    * tensor_dimension_params.src[3].nb[0]
         + y * tensor_dimension_params.src[3].nb[1] +
         z * tensor_dimension_params.src[3].nb[2]
            // + tensor_dimension_params.src[3].offset
             ] = v;
}

fn set_src4(x: u32, y: u32, z: u32, v: f16) {
    src4[ x 
        //    * tensor_dimension_params.src[4].nb[0]
         + y * tensor_dimension_params.src[4].nb[1] +
         z * tensor_dimension_params.src[4].nb[2]
            // + tensor_dimension_params.src[4].offset
             ] = v;
}

fn set_src5(x: u32, y: u32, z: u32, v: f16) {
    src5[ x 
        //    * tensor_dimension_params.src[5].nb[0]
         + y * tensor_dimension_params.src[5].nb[1] +
         z * tensor_dimension_params.src[5].nb[2]
            // + tensor_dimension_params.src[5].offset
             ] = v;
}


fn num_el_dst() -> u32 {
    let ne0 = tensor_dimension_params.dst.nb[1] / tensor_dimension_params.dst.nb[0];
    return ne0 * u32(tensor_dimension_params.dst.ne[1] * tensor_dimension_params.dst.ne[2] * tensor_dimension_params.dst.ne[3]);
    // return u32(tensor_dimension_params.dst.ne[0] * tensor_dimension_params.dst.ne[1] * tensor_dimension_params.dst.ne[2] * tensor_dimension_params.dst.ne[3]);
}

);

static const char src_ggml_shader_kernel_silu[] = MULTILINE(

@compute
@workgroup_size(1)
fn kernel_silu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x =  get_src0_lin(global_id.x);
    set_dst_lin(global_id.x, x / (1.0 + exp(-x)));
}

);

static const char src_ggml_shader_kernel_conv_1d_small_kern[] = MULTILINE(

@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let s0 = u32(tensor_dimension_params.params[0][0]);
    let p0 = u32(tensor_dimension_params.params[0][1]);
    let d0 = u32(tensor_dimension_params.params[0][2]);
    let apply_tanh = bool(tensor_dimension_params.params[0][3]);
    let has_bias = bool(tensor_dimension_params.params[1][0]);
    let has_inject_signal = bool(tensor_dimension_params.params[1][1]);
    let nk = u32(tensor_dimension_params.src[0].ne[2]);


    let input_channels = u32(tensor_dimension_params.src[0].ne[1]);
    let output_channels = u32(tensor_dimension_params.dst.ne[1]);
    let input_len = u32(tensor_dimension_params.src[1].ne[0]);
    let output_len = u32(tensor_dimension_params.dst.ne[0]);
    let num_batches = u32(tensor_dimension_params.dst.ne[2]);

    if (global_id.x >= output_len) {
        return;
    }
    if (global_id.y >= output_channels) {
        return;
    }
    if (global_id.z >= num_batches) {
        return;
    }

    let real_input_len = s0*(output_len - 1u) + d0*(nk - 1u) + 1u - 2u*p0;

    var output : f32 = 0.0;

    if (has_bias) {
        // let bias_idx = global_id.y * tensor_dimension_params.src[2].nb[1];
        // let bias = f16(extra_uniform1[bias_idx/4u][bias_idx%4u]);
        let bias = get_src2(0u, global_id.y, 0u);
        output += f32(bias);
    }

    if (has_inject_signal) {
        output += f32(get_src3(u32(tensor_dimension_params.src[3].ne[0]) - output_len + global_id.x, global_id.y, global_id.z));
    }

    let base_src1_offset = input_len - real_input_len + global_id.x + global_id.z * tensor_dimension_params.src[1].nb[2];

    for (var ik = 0u; ik < nk; ik = ik + 1u) {
        let in_idx_offset = ik * d0 + base_src1_offset;
        let kernel_base_idx = global_id.y + ik * tensor_dimension_params.src[0].nb[2];
        for (var ic = 0u; ic < input_channels; ic = ic + 1u) {
            let input = get_src1_lin(in_idx_offset + ic * tensor_dimension_params.src[1].nb[1]);
            let kernel = get_src0(global_id.y, ic, ik);
            // let kernel_idx = kernel_base_idx + ic * tensor_dimension_params.src[0].nb[1];
            // let kernel = f16(extra_uniform0[kernel_idx/4u][kernel_idx%4u]);
            output = output + f32(input) * f32(kernel);
        }
    }

    if (apply_tanh) {
        output = tanh(output);
    }

    set_dst(global_id.x, global_id.y, global_id.z, f16(output));
}

);

static const char src_ggml_shader_kernel_conv_1d_small_kern_no_offsets[] = MULTILINE(

@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern_no_offsets(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let d0 = u32(tensor_dimension_params.params[0][2]);
    let apply_tanh = bool(tensor_dimension_params.params[0][3]);
    let has_bias = bool(tensor_dimension_params.params[1][0]);
    let has_inject_signal = bool(tensor_dimension_params.params[1][1]);
    let nk = u32(tensor_dimension_params.src[0].ne[2]);


    let input_channels = u32(tensor_dimension_params.src[0].ne[1]);
    let output_channels = u32(tensor_dimension_params.dst.ne[1]);
    let output_len = u32(tensor_dimension_params.dst.ne[0]);

    let mult_idx = global_id.x * 4u;

    if (mult_idx >= output_len) {
        return;
    }
    if (global_id.y >= output_channels) {
        return;
    }

    // var output : f16 = 0.0;
    var output = vec4f();

    if (has_bias) {
        // let bias_idx = global_id.y * tensor_dimension_params.src[2].nb[1];
        // let bias = extra_uniform1[bias_idx/4u][bias_idx%4u];
        let bias = get_src2(0u, global_id.y, 0u);
        output += f32(bias);
    }

    if (has_inject_signal) {
        let src3_idx = global_id.x + global_id.y * tensor_dimension_params.src[3].nb[1]/4u + global_id.z * tensor_dimension_params.src[3].nb[2]/4u;
        output += vec4f(src3_v4[src3_idx]);
        // output.x += get_src3(mult_idx,      global_id.y, global_id.z);
        // output.y += get_src3(mult_idx + 1u, global_id.y, global_id.z);
        // output.z += get_src3(mult_idx + 2u, global_id.y, global_id.z);
        // output.w += get_src3(mult_idx + 3u, global_id.y, global_id.z);
    }

    let base_src1_offset = mult_idx + global_id.z * tensor_dimension_params.src[1].nb[2];

    for (var ik = 0u; ik < nk; ik = ik + 1u) {
        let in_idx_offset = ik * d0 + base_src1_offset;
        // let kernel_base_idx = global_id.y + ik * tensor_dimension_params.src[0].nb[2];
        for (var ic = 0u; ic < input_channels; ic = ic + 1u) {
            // let input_idx = in_idx_offset + ic * tensor_dimension_params.src[1].nb[1];
            let input_idx = (in_idx_offset + ic * tensor_dimension_params.src[1].nb[1])/4u;
            var input = src1_v4[input_idx];
            // input.x = get_src1_lin(input_idx);
            // input.y = get_src1_lin(input_idx+1u);
            // input.z = get_src1_lin(input_idx+2u);
            // input.w = get_src1_lin(input_idx+3u);
            let kernel = get_src0(global_id.y, ic, ik);
            // let kernel_idx = kernel_base_idx + ic * tensor_dimension_params.src[0].nb[1];
            // let kernel = extra_uniform0[kernel_idx/4u][kernel_idx%4u];
            output = output + vec4f(input) * f32(kernel);
        }
    }

    if (apply_tanh) {
        output = tanh(output);
    }

    let dst_idx = global_id.x + global_id.y * tensor_dimension_params.dst.nb[1]/4u + global_id.z * tensor_dimension_params.dst.nb[2]/4u;
    dst_v4[dst_idx] = vec4h(output);

    // set_dst(mult_idx, global_id.y, global_id.z,    output.x);
    // set_dst(mult_idx+1u, global_id.y, global_id.z, output.y);
    // set_dst(mult_idx+2u, global_id.y, global_id.z, output.z);
    // set_dst(mult_idx+3u, global_id.y, global_id.z, output.w);
}

);

// const nk_8x8x3 = 3u;
// const channels_8x8x3 = 8u;
// var<workgroup> workgroup_data_input_8x8x3: array<array<vec4h, channels_8x8x3>, 32>;
// @compute
// @workgroup_size(32, channels_8x8x3)
// fn kernel_conv_1d_small_kern_no_offsets_8x8x3(@builtin(global_invocation_id) global_id: vec3<u32>, 
//     @builtin(workgroup_id) wg_id: vec3<u32>,
//     @builtin(local_invocation_id) local_id: vec3<u32>) {
//     let d0 = u32(tensor_dimension_params.params[0][2]);
//     let d0d4 = d0 / 4u;
//     let apply_tanh = bool(tensor_dimension_params.params[0][3]);
//     let has_bias = bool(tensor_dimension_params.params[1][0]);
//     let has_inject_signal = bool(tensor_dimension_params.params[1][1]);

//     let output_len = u32(tensor_dimension_params.dst.ne[0]);
//     let output_len_d4 = (output_len + 3u) / 4u;

//     let kern_output_vec_values_per_thread = 16u;
//     let start_idx_d4 = (global_id.x/d0d4) * kern_output_vec_values_per_thread * d0d4 + (global_id.x) % d0d4;

//     if (start_idx_d4 >= output_len_d4) {
//         return;
//     }

//     var kernel = array<array<f16, channels_8x8x3>, nk_8x8x3>();
//     for (var ik = 0u; ik < nk_8x8x3; ik = ik + 1u) {
//         for (var ic = 0u; ic < channels_8x8x3; ic = ic + 1u) {
//             kernel[ik][ic] = get_src0(local_id.y, ic, ik);
//         }
//     }

//     let values_vec_this_thread = min((output_len_d4 - start_idx_d4) / d0d4 + 1u, kern_output_vec_values_per_thread);
//     let input_values_vec_this_thread = values_vec_this_thread + nk_8x8x3 - 1u;


//     var output = array<vec4h, nk_8x8x3>();

//     var bias = 0.0;
//     if (has_bias) {
//         bias = get_src2(0u, local_id.y, 0u);
//     }

//     let input_base_idx = start_idx_d4 + (local_id.y * tensor_dimension_params.src[1].nb[1] + global_id.z * tensor_dimension_params.src[1].nb[2]) / 4u;
//     let src3_base_idx  = start_idx_d4 + (local_id.y * tensor_dimension_params.src[3].nb[1] + global_id.z * tensor_dimension_params.src[3].nb[2]) / 4u;
//     let dst_base_idx   = start_idx_d4 + (local_id.y * tensor_dimension_params.dst.nb[1]    + global_id.z * tensor_dimension_params.dst.nb[2]   ) / 4u;

//     for (var i = 0u; i < input_values_vec_this_thread; i = i + 1u) {
//         workgroup_data_input_8x8x3[local_id.x][local_id.y] = src1_v4[input_base_idx + i * d0d4];
//         workgroupBarrier();

//         var src3_here = vec4h();
//         let dst_offs_vec_idx = (max(i, nk_8x8x3 - 1u) + 1u - nk_8x8x3) * d0d4;
//         if (has_inject_signal && i >= (nk_8x8x3 - 1u)) {
//             src3_here = src3_v4[src3_base_idx + dst_offs_vec_idx];
//         }

//         for (var ic = 0u; ic < channels_8x8x3; ic = ic + 1u) {
//             let ic_adj = (ic + local_id.y) % channels_8x8x3;
//             let in_here = workgroup_data_input_8x8x3[local_id.x][ic_adj];
//             for (var ik = 0u; ik < nk_8x8x3; ik = ik + 1u) {
//                 let kern_idx = (i + ik) % nk_8x8x3;
//                 output[ik] = output[ik] + in_here * kernel[kern_idx][ic_adj];
//             }
//         }

//         let reg_idx = nk_8x8x3 - 1u - (i % nk_8x8x3);

//         if (i >= (nk_8x8x3 - 1u)) {
//             output[reg_idx] = output[reg_idx] + bias;
//             if (has_inject_signal) {
//                 output[reg_idx] += src3_here;
//             }
//             if (apply_tanh) {
//                 output[reg_idx] = tanh(output[reg_idx]);
//             }
//             dst_v4[dst_base_idx + dst_offs_vec_idx] = output[reg_idx];
//         }

//         output[reg_idx] = vec4h();
//     }
// }


// const kernel_conv_1d_small_kern_output_values_per_thread = 16u;
// const kernel_conv_1d_small_kern_output_channels_per_warp = 16u;
// const kernel_conv_1d_small_kern_num_threads_x = 16u;
// const kernel_conv_1d_small_kern_input_channels = 16u;
// const kernel_conv_1d_small_kern_nk = 3u;
// const kernel_conv_1d_small_kern_input_values_per_thread = kernel_conv_1d_small_kern_output_values_per_thread + kernel_conv_1d_small_kern_nk - 1u;
// const kernel_conv_1d_small_kern_total_kernel_size = kernel_conv_1d_small_kern_output_channels_per_warp * kernel_conv_1d_small_kern_input_channels * kernel_conv_1d_small_kern_nk;
// // const kernel_conv_1d_small_kern_total_input_size = kernel_conv_1d_small_kern_input_channels * kernel_conv_1d_small_kern_input_values_per_thread * kernel_conv_1d_small_kern_num_threads_x;
// const total_kernel_invocs_warp = kernel_conv_1d_small_kern_output_channels_per_warp * kernel_conv_1d_small_kern_num_threads_x;
// const iters_to_load_kernel = (kernel_conv_1d_small_kern_total_kernel_size + total_kernel_invocs_warp - 1u) / total_kernel_invocs_warp;

// var<workgroup> workgroup_data_kernel: array<array<array<f16, kernel_conv_1d_small_kern_input_channels>, kernel_conv_1d_small_kern_nk>, kernel_conv_1d_small_kern_output_channels_per_warp>;
// var<workgroup> workgroup_data_input:  array<array<f16, kernel_conv_1d_small_kern_num_threads_x>, kernel_conv_1d_small_kern_input_channels>;


// fn get_dilated_start_idx(x: u32, d0: u32) -> u32 {
//     return (x/d0) * kernel_conv_1d_small_kern_output_values_per_thread * d0 + x % d0;
// }

// fn get_dilated_idx(start_idx: u32, idx: u32, d0: u32) -> u32 {
//     return start_idx + idx * d0;
// }

// fn index_linear_from_3d(x: u32, y: u32, z: u32, x_max:u32, y_max: u32, z_max: u32) -> u32 {
//     return (z * y_max + y) * x_max + x;
// }

// fn index_3d_from_linear(idx: u32, x_max:u32, y_max: u32, z_max: u32) -> vec3<u32> {
//     let z = idx / (x_max * y_max);
//     let y = (idx - z * x_max * y_max) / x_max;
//     let x = idx - z * x_max * y_max - y * x_max;
//     return vec3<u32>(x, y, z);
// }

// @compute
// @workgroup_size(kernel_conv_1d_small_kern_num_threads_x, kernel_conv_1d_small_kern_output_channels_per_warp)
// fn kernel_conv_1d_small_kern_opti(@builtin(global_invocation_id) global_id: vec3<u32>, 
//     @builtin(workgroup_id) wg_id: vec3<u32>,
//     @builtin(local_invocation_id) local_id: vec3<u32>,
//     @builtin(local_invocation_index) local_index: u32) {
//     let s0 = u32(tensor_dimension_params.params[0][0]);
//     let p0 = u32(tensor_dimension_params.params[0][1]);
//     let d0 = u32(tensor_dimension_params.params[0][2]);
//     let apply_tanh = bool(tensor_dimension_params.params[0][3]);
//     let has_bias = bool(tensor_dimension_params.params[1][0]);
//     let has_inject_signal = bool(tensor_dimension_params.params[1][1]);
//     let nk = u32(tensor_dimension_params.src[0].ne[2]);


//     let input_channels = u32(tensor_dimension_params.src[0].ne[1]);
//     let output_channels = u32(tensor_dimension_params.dst.ne[1]);
//     let input_len = u32(tensor_dimension_params.src[1].ne[0]);
//     let output_len = u32(tensor_dimension_params.dst.ne[0]);
//     let num_batches = u32(tensor_dimension_params.dst.ne[2]);

//     let start_idx = get_dilated_start_idx(global_id.x, d0);

//     if (start_idx >= output_len) {
//         return;
//     }
//     if (global_id.y >= output_channels) {
//         return;
//     }
//     // if (global_id.z >= num_batches) {
//     //     return;
//     // }

//     let values_this_thread = min((output_len - start_idx) / d0 + 1u, kernel_conv_1d_small_kern_output_values_per_thread);
//     let input_values_this_thread = values_this_thread + kernel_conv_1d_small_kern_nk - 1u;

//     let real_input_len = s0*(output_len - 1u) + d0*(nk - 1u) + 1u - 2u*p0;

//     var output = array<f16, kernel_conv_1d_small_kern_output_values_per_thread>();

//     if (has_bias) {
//         let ph = get_src2(0u, global_id.y, 0u);
//         for (var i = 0u; i < values_this_thread; i = i + 1u) {
//             output[i] = ph;
//         }
//     }

//     if (has_inject_signal) {
//         for (var i = 0u; i < values_this_thread; i = i + 1u) {
//             output[i] += get_src3(u32(tensor_dimension_params.src[3].ne[0]) - output_len + get_dilated_idx(start_idx, i, d0), global_id.y, global_id.z);
//         }
//     }

//     let base_in_idx_offset = input_len - real_input_len;

//     for (var i=0u; i<input_values_this_thread; i=i+1u) {
//         for (var ic = 0u; ic < input_channels; ic = ic + 1u) {
//             let input = get_src1(base_in_idx_offset + get_dilated_idx(start_idx, i, d0), ic, global_id.z);
//             for (var ik = 0u; ik < nk; ik = ik + 1u) {
//                 let iout = i32(i + ik) - 2;
//                 if ( iout >= 0 && iout < i32(values_this_thread)) {
//                     let kernel = get_src0(global_id.y, ic, nk-ik- 1u);
//                     output[iout] = output[iout] + input * kernel;
//                 }
//             }
//         }
//     }

//     if (apply_tanh) {
//         for (var i = 0u; i < values_this_thread; i = i + 1u) {
//             output[i] = tanh(output[i]);
//         }
//     }

//     for (var i = 0u; i < values_this_thread; i = i + 1u) {
//         set_dst(get_dilated_idx(start_idx, i, d0), global_id.y, global_id.z, output[i]);
//     }
// }


// @compute
// @workgroup_size(kernel_conv_1d_small_kern_num_threads_x, kernel_conv_1d_small_kern_output_channels_per_warp)
// fn kernel_conv_1d_small_kern_opti_large_dil(@builtin(global_invocation_id) global_id: vec3<u32>, 
//     @builtin(workgroup_id) wg_id: vec3<u32>,
//     @builtin(local_invocation_id) local_id: vec3<u32>,
//     @builtin(local_invocation_index) local_index: u32) {
//     let s0 = u32(tensor_dimension_params.params[0][0]);
//     let p0 = u32(tensor_dimension_params.params[0][1]);
//     let d0 = u32(tensor_dimension_params.params[0][2]);
//     let apply_tanh = bool(tensor_dimension_params.params[0][3]);
//     let has_bias = bool(tensor_dimension_params.params[1][0]);
//     let has_inject_signal = bool(tensor_dimension_params.params[1][1]);
//     let nk = u32(tensor_dimension_params.src[0].ne[2]);


//     let input_channels = u32(tensor_dimension_params.src[0].ne[1]);
//     let output_channels = u32(tensor_dimension_params.dst.ne[1]);
//     let input_len = u32(tensor_dimension_params.src[1].ne[0]);
//     let output_len = u32(tensor_dimension_params.dst.ne[0]);
//     let num_batches = u32(tensor_dimension_params.dst.ne[2]);

//     let start_idx = get_dilated_start_idx(wg_id.x*kernel_conv_1d_small_kern_num_threads_x, d0);

//     if (start_idx >= output_len) {
//         return;
//     }
//     // if (global_id.y >= output_channels) {
//     //     return;
//     // }
//     // if (global_id.z >= num_batches) {
//     //     return;
//     // }

//     for (var ik=0u; ik<kernel_conv_1d_small_kern_nk; ik=ik+1u) {
//         workgroup_data_kernel[local_id.y][nk-ik- 1u][local_id.x] = get_src0(local_id.x, local_id.y, ik);
//     }
//     workgroupBarrier();

//     let values_this_thread = min((output_len - start_idx) / d0 + 1u, kernel_conv_1d_small_kern_output_values_per_thread);
//     let input_values_this_thread = values_this_thread + kernel_conv_1d_small_kern_nk - 1u;

//     let real_input_len = s0*(output_len - 1u) + d0*(nk - 1u) + 1u - 2u*p0;

//     var output = array<f16, kernel_conv_1d_small_kern_output_values_per_thread>();

//     if (has_bias) {
//         let ph = get_src2(0u, global_id.y, 0u);
//         for (var i = 0u; i < values_this_thread; i = i + 1u) {
//             output[i] = ph;
//         }
//     }

//     if (has_inject_signal) {
//         let offs1 = local_id.x + u32(tensor_dimension_params.src[3].ne[0]) - output_len + start_idx + global_id.y * tensor_dimension_params.src[3].nb[1] +
//             global_id.z * tensor_dimension_params.src[3].nb[2];
//         for (var i = 0u; i < values_this_thread; i = i + 1u) {
//             // output[i] += get_src3(u32(tensor_dimension_params.src[3].ne[0]) - output_len + start_idx + local_id.x + i*d0, global_id.y, global_id.z);
//             output[i] += get_src3_lin(i*d0+offs1);
//         }
//     }

//     let base_in_idx_offset = input_len - real_input_len;
//     let base_in_idx = base_in_idx_offset + start_idx + local_id.x + global_id.z * tensor_dimension_params.src[1].nb[2];

//     for (var i=0u; i<input_values_this_thread; i=i+1u) {
//         workgroup_data_input[local_id.y][local_id.x] = get_src1_lin(base_in_idx + i*d0 + local_id.y * tensor_dimension_params.src[1].nb[1]);
//         workgroupBarrier();
//         for (var ic = 0u; ic < kernel_conv_1d_small_kern_input_channels; ic = ic + 1u) {
//             // let input = get_src1_lin(base_in_idx + i*d0 + ic * tensor_dimension_params.src[1].nb[1]);
//             let input = workgroup_data_input[ic][local_id.x];
//             for (var ik = 0u; ik < kernel_conv_1d_small_kern_nk; ik = ik + 1u) {
//                 let iout = i32(i + ik) - 2;
//                 if ( iout >= 0 && iout < i32(values_this_thread)) {
//                     let kernel = workgroup_data_kernel[ic][ik][global_id.y];
//                     // let kernel = get_src0(global_id.y, ic, nk-ik- 1u);
//                     output[iout] = output[iout] + input * kernel;
//                 }
//             }
//         }
//     }

//     if (apply_tanh) {
//         for (var i = 0u; i < values_this_thread; i = i + 1u) {
//             output[i] = tanh(output[i]);
//         }
//     }

//     let offs1 = local_id.x + start_idx + global_id.y * tensor_dimension_params.dst.nb[1] +
//             global_id.z * tensor_dimension_params.dst.nb[2];
//     for (var i = 0u; i < values_this_thread; i = i + 1u) {
//         set_dst_lin(i*d0+offs1, output[i]);
//         // set_dst(start_idx + local_id.x + i*d0, global_id.y, global_id.z, output[i]);
//     }
// }

static const char src_ggml_shader_kernel_conv_1d_small_kern_simpl[] = MULTILINE(

@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern_simpl(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let has_bias = bool(tensor_dimension_params.params[1][0]);
    let has_inject_signal = bool(tensor_dimension_params.params[1][1]);


    let input_channels = u32(tensor_dimension_params.src[0].ne[1]);
    let input_len = u32(tensor_dimension_params.src[1].ne[0]);
    let output_len = u32(tensor_dimension_params.dst.ne[0]);

    if (global_id.x >= output_len) {
        return;
    }

    var output : f32 = 0.0;

    if (has_bias) {
        output += f32(get_src2(0u, global_id.y, 0u));
    }

    if (has_inject_signal) {
        output += f32(get_src3(u32(tensor_dimension_params.src[3].ne[0]) - output_len + global_id.x, global_id.y, global_id.z));
    }

    let in_idx_offset = input_len - output_len + global_id.x;
    for (var ic = 0u; ic < input_channels; ic = ic + 1u) {
        let input = get_src1(in_idx_offset, ic, global_id.z);
        let kernel = get_src0(global_id.y, ic, 0u);
        output = output + f32(input) * f32(kernel);
    }

    set_dst(global_id.x, global_id.y, global_id.z, f16(output));
}

);


static const char src_ggml_shader_kernel_add_and_trim[] = MULTILINE(

@compute
@workgroup_size(256)
fn kernel_add_and_trim(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_len = u32(tensor_dimension_params.dst.ne[0]);

    if (global_id.x >= output_len) {
        return;
    }
    
    set_dst(global_id.x, global_id.y, global_id.z, 
        get_src0(global_id.x + u32(tensor_dimension_params.src[0].ne[0]) - output_len, global_id.y, global_id.z) + 
        get_src1(global_id.x + u32(tensor_dimension_params.src[1].ne[0]) - output_len, global_id.y, global_id.z)
    );
}

);

static const char src_ggml_shader_kernel_scale[] = MULTILINE(


@compute
@workgroup_size(256)
fn kernel_scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_el_dst()) {
        return;
    }
    set_dst_lin(global_id.x, get_src0_lin(global_id.x) * get_src1_lin(0u));
}

);


static const char src_ggml_shader_kernel_scale_inplace[] = MULTILINE(

@compute
@workgroup_size(256)
fn kernel_scale_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_el_dst()) {
        return;
    }
    set_dst_lin(global_id.x, get_dst_lin(global_id.x) * get_src1_lin(0u));
}

);

static const char src_ggml_shader_kernel_sub[] = MULTILINE(


@compute
@workgroup_size(256)
fn kernel_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_el_dst()) {
        return;
    }
    if ((global_id.x % tensor_dimension_params.dst.nb[1])<u32(tensor_dimension_params.dst.ne[0])) {
        set_dst_lin(global_id.x, get_src0_lin(global_id.x) - get_src1_lin(global_id.x));
    } else {
        set_dst_lin(global_id.x, 0.0);
    }
    // set_dst_lin(global_id.x, get_src0_lin(global_id.x) - get_src1_lin(global_id.x));
}

);

static const char src_ggml_shader_kernel_sqr[] = MULTILINE(


@compute
@workgroup_size(256)
fn kernel_sqr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_el_dst()) {
        return;
    }
    let x = get_src0_lin(global_id.x);
    set_dst_lin(global_id.x, x * x);
}

);


static const char src_ggml_shader_kernel_sum[] = MULTILINE(

var<workgroup> workgroup_data: array<f32, 256>;

@compute
@workgroup_size(256)
fn kernel_sum(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let ne00 = tensor_dimension_params.src[0].nb[1] / tensor_dimension_params.src[0].nb[0];
    let num_el_src0 = ne00 * u32(tensor_dimension_params.src[0].ne[1] * tensor_dimension_params.src[0].ne[2] * tensor_dimension_params.src[0].ne[3]);

    var sum : f32 = 0.0;
    
    for (var i = local_id.x; i < num_el_src0; i = i + 256u) {
        sum = sum + f32(get_src0_lin(i));
    }

    workgroup_data[local_id.x] = sum;
    workgroupBarrier();

    if (0u == local_id.x) {
        sum = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            sum = sum + workgroup_data[i];
        }
        set_dst_lin(0u, f16(sum));
    }
}

);


static const char src_ggml_shader_kernel_repeat[] = MULTILINE(


@compute
@workgroup_size(256)
fn kernel_repeat(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(tensor_dimension_params.dst.ne[0])) {
        return;
    }

    let idx0 = global_id.x * u32(tensor_dimension_params.src[0].ne[0]) / u32(tensor_dimension_params.dst.ne[0]);
    let idx1 = global_id.y * u32(tensor_dimension_params.src[0].ne[1]) / u32(tensor_dimension_params.dst.ne[1]);
    let idx2 = global_id.z * u32(tensor_dimension_params.src[0].ne[2]) / u32(tensor_dimension_params.dst.ne[2]);

    set_dst(global_id.x, global_id.y, global_id.z, get_src0(idx0, idx1, idx2));
}

);


static const char src_ggml_shader_kernel_mul[] = MULTILINE(


@compute
@workgroup_size(256)
fn kernel_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(tensor_dimension_params.dst.ne[0])) {
        return;
    }

    let idx0 = global_id.x * u32(tensor_dimension_params.src[1].ne[0]) / u32(tensor_dimension_params.dst.ne[0]);
    let idx1 = global_id.y * u32(tensor_dimension_params.src[1].ne[1]) / u32(tensor_dimension_params.dst.ne[1]);
    let idx2 = global_id.z * u32(tensor_dimension_params.src[1].ne[2]) / u32(tensor_dimension_params.dst.ne[2]);

    set_dst(global_id.x, global_id.y, global_id.z, 
        get_src0(global_id.x, global_id.y, global_id.z) * get_src1(idx0, idx1, idx2));
}


);


static const char src_ggml_shader_kernel_conv_1d_small_kern_back_filter[] = MULTILINE(

var<workgroup> workgroup_data: array<f32, 256>;

@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern_back_filter(@builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let s0 = u32(tensor_dimension_params.params[0][0]);
    let p0 = u32(tensor_dimension_params.params[0][1]);
    let d0 = u32(tensor_dimension_params.params[0][2]);
    let nk = u32(tensor_dimension_params.dst.ne[2]);


    let input_channels = u32(tensor_dimension_params.src[0].ne[1]);
    let output_channels = u32(tensor_dimension_params.dst.ne[0]);
    let input_len = u32(tensor_dimension_params.src[0].ne[0]);
    let output_len = u32(tensor_dimension_params.src[1].ne[0]);
    let num_batches = u32(tensor_dimension_params.src[0].ne[2]);

    // if (wg_id.x >= nk) {
    //     return;
    // }
    // if (global_id.z >= input_channels) {
    //     return;
    // }
    // if (global_id.y >= output_channels) {
    //     return;
    // }

    let real_input_len = s0*(output_len - 1u) + d0*(nk - 1u) + 1u - 2u*p0;

    var output : f32 = 0.0;

    let base_offset = wg_id.x * d0 + input_len - real_input_len;

    for (var ir = 0u; ir < num_batches; ir = ir + 1u) {
        let base_idx_src0 = base_offset + ir * tensor_dimension_params.src[0].nb[2] + global_id.z * tensor_dimension_params.src[0].nb[1];
        let base_idx_src1 = ir * tensor_dimension_params.src[1].nb[2] + global_id.y * tensor_dimension_params.src[1].nb[1];
        for (var isample = local_id.x; isample < output_len; isample = isample + 256u) {
            // output = output + 
            //     get_src0(base_offset + isample, global_id.z, ir) * get_src1(isample, global_id.y, ir);
            output = output + f32(get_src0_lin(base_idx_src0 + isample)) * f32(get_src1_lin(base_idx_src1 + isample));
        }
        // workgroupBarrier();
    }

    workgroup_data[local_id.x] = output;
    workgroupBarrier();

    if (0u == local_id.x) {
        output = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            output = output + workgroup_data[i];
        }

        set_dst(global_id.y, global_id.z, wg_id.x, f16(output));
    }
}

);


static const char src_ggml_shader_kernel_conv_1d_small_kern_back_filter_nk1[] = MULTILINE(

var<workgroup> workgroup_data: array<f32, 256>;

@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern_back_filter_nk1(@builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let input_len = u32(tensor_dimension_params.src[0].ne[0]);
    let output_len = u32(tensor_dimension_params.src[1].ne[0]);
    let num_batches = u32(tensor_dimension_params.src[0].ne[2]);

    var output : f32 = 0.0;
    let base_offset = input_len - output_len + global_id.y * tensor_dimension_params.src[0].nb[1];

    for (var ir = 0u; ir < num_batches; ir = ir + 1u) {
        let base_idx_src0 = base_offset + ir * tensor_dimension_params.src[0].nb[2];
        let base_idx_src1 = ir * tensor_dimension_params.src[1].nb[2] + wg_id.x * tensor_dimension_params.src[1].nb[1];
        for (var isample = local_id.x; isample < output_len; isample = isample + 256u) {
            output = output + f32(get_src0_lin(base_idx_src0 + isample)) * f32(get_src1_lin(base_idx_src1 + isample));
        }
    }

    workgroup_data[local_id.x] = output;
    workgroupBarrier();

    if (0u == local_id.x) {
        output = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            output = output + workgroup_data[i];
        }

        set_dst(wg_id.x, global_id.y, 0u, f16(output));
    }
}

);



static const char src_ggml_shader_kernel_conv_1d_small_kern_back_filter_stage1[] = MULTILINE(

@group(0) @binding(5)
var<storage,read_write> src5_f32: array<f32>;

var<workgroup> workgroup_data: array<f32, 256>;

@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern_back_filter_stage1(@builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let s0 = u32(tensor_dimension_params.params[0][0]);
    let p0 = u32(tensor_dimension_params.params[0][1]);
    let d0 = u32(tensor_dimension_params.params[0][2]);
    let nk = u32(tensor_dimension_params.dst.ne[2]);


    let input_channels = u32(tensor_dimension_params.src[0].ne[1]);
    let output_channels = u32(tensor_dimension_params.dst.ne[0]);
    let input_len = u32(tensor_dimension_params.src[0].ne[0]);
    let output_len = u32(tensor_dimension_params.src[1].ne[0]);
    let num_batches = u32(tensor_dimension_params.src[0].ne[2]);

    let idx_ic = global_id.z / num_batches;
    let idx_ir = global_id.z % num_batches;
    let idx_ik = wg_id.x;

    let real_input_len = s0*(output_len - 1u) + d0*(nk - 1u) + 1u - 2u*p0;

    var output : f32 = 0.0;

    let base_offset = idx_ik * d0 + input_len - real_input_len;

    let base_idx_src0 = base_offset + idx_ir * tensor_dimension_params.src[0].nb[2] + idx_ic * tensor_dimension_params.src[0].nb[1];
    let base_idx_src1 = idx_ir * tensor_dimension_params.src[1].nb[2] + global_id.y * tensor_dimension_params.src[1].nb[1];
    for (var isample = local_id.x; isample < output_len; isample = isample + 256u) {
        output = output + f32(get_src0_lin(base_idx_src0 + isample)) * f32(get_src1_lin(base_idx_src1 + isample));
    }

    workgroup_data[local_id.x] = output;
    workgroupBarrier();

    if (0u == local_id.x) {
        output = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            output = output + workgroup_data[i];
        }

        let nb1 = num_batches;
        let nb2 = nb1 * output_channels;
        let nb3 = nb2 * input_channels;

        src5_f32[idx_ir + nb1 * global_id.y + nb2 * idx_ic +  nb3 * idx_ik] = output;
    }
}

);


static const char src_ggml_shader_kernel_conv_1d_small_kern_back_filter_stage2[] = MULTILINE(

@group(0) @binding(5)
var<storage,read_write> src5_f32: array<f32>;

@compute
@workgroup_size(1)
fn kernel_conv_1d_small_kern_back_filter_stage2(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let num_batches = u32(tensor_dimension_params.src[0].ne[2]);
    let input_channels = u32(tensor_dimension_params.src[0].ne[1]);
    let output_channels = u32(tensor_dimension_params.dst.ne[0]);

    var output : f32 = 0.0;

    let nb1 = num_batches;
    let nb2 = nb1 * output_channels;
    let nb3 = nb2 * input_channels;

    let base_idx_src1 = nb1 * global_id.y + nb2 * global_id.z +  nb3 * wg_id.x;
    for (var isample = 0u; isample < num_batches; isample = isample + 1u) {
        output = output + src5_f32[base_idx_src1 + isample];
    }
    set_dst(global_id.y, global_id.z, wg_id.x, f16(output));
}

);


static const char src_ggml_shader_kernel_conv_1d_small_kern_back_input[] = MULTILINE(


@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern_back_input(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s0 = u32(tensor_dimension_params.params[0][0]);
    let p0 = u32(tensor_dimension_params.params[0][1]);
    let d0 = u32(tensor_dimension_params.params[0][2]);
    let accumulate = bool(tensor_dimension_params.params[0][3]);
    let nk = u32(tensor_dimension_params.src[0].ne[2]);

    let output_channels = u32(tensor_dimension_params.src[0].ne[0]);
    let input_len = u32(tensor_dimension_params.dst.ne[0]);
    let output_len = u32(tensor_dimension_params.src[1].ne[0]);

    if (global_id.x >= input_len) {
        return;
    }

    var output : f32 = 0.0;

    if (accumulate) {
        output = f32(get_src2(global_id.x, global_id.y, global_id.z));
    }

    for (var ik = 0u; ik < nk; ik = ik + 1u) {
        let idx_offset = ik * d0;
        if (global_id.x >= idx_offset && (global_id.x < (idx_offset+output_len))) {
            let base_idx_src0 = global_id.y * tensor_dimension_params.src[0].nb[1] + ik * tensor_dimension_params.src[0].nb[2];
            let base_idx_src1 = global_id.z * tensor_dimension_params.src[1].nb[2] + (global_id.x - idx_offset);
            for (var idx_oc = 0u; idx_oc < output_channels; idx_oc = idx_oc + 1u) {
                // output = output + 
                //     get_src0(idx_oc, global_id.y, ik) * 
                //     get_src1(global_id.x - idx_offset, idx_oc, global_id.z);
                output = output + f32(get_src0_lin(base_idx_src0 + idx_oc)) * f32(get_src1_lin(base_idx_src1 + idx_oc * tensor_dimension_params.src[1].nb[1]));
            }
        }
    }

    set_dst(global_id.x, global_id.y, global_id.z, f16(output));
}

);


static const char src_ggml_shader_kernel_conv_1d_small_kern_back_input_large_dil[] = MULTILINE(

@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern_back_input_large_dil(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s0 = u32(tensor_dimension_params.params[0][0]);
    let p0 = u32(tensor_dimension_params.params[0][1]);
    let d0 = u32(tensor_dimension_params.params[0][2]);
    let accumulate = bool(tensor_dimension_params.params[0][3]);
    let nk = u32(tensor_dimension_params.src[0].ne[2]);

    let output_channels = u32(tensor_dimension_params.src[0].ne[0]);
    let input_len = u32(tensor_dimension_params.dst.ne[0]);
    let output_len = u32(tensor_dimension_params.src[1].ne[0]);

    let mult_idx = global_id.x * 4u;

    if (mult_idx >= input_len) {
        return;
    }


    var output = vec4f();

    if (accumulate) {
        let idx_src2 = (global_id.z * tensor_dimension_params.src[2].nb[2] +
            global_id.y * tensor_dimension_params.src[2].nb[1] +
            mult_idx) / 4u;
        output = vec4f(src2_v4[idx_src2]);
    }

    for (var ik = 0u; ik < nk; ik = ik + 1u) {
        let idx_offset = ik * d0;
        if (mult_idx >= idx_offset && (mult_idx < (idx_offset+output_len))) {
            let base_idx_src0 = global_id.y * tensor_dimension_params.src[0].nb[1] + ik * tensor_dimension_params.src[0].nb[2];
            let base_idx_src1 = global_id.z * tensor_dimension_params.src[1].nb[2] + (mult_idx - idx_offset);
            let mult1 = vec4f(
                f32((mult_idx - idx_offset) < output_len),
                f32((mult_idx - idx_offset + 1u) < output_len),
                f32((mult_idx - idx_offset + 2u) < output_len),
                f32((mult_idx - idx_offset + 3u) < output_len)
            );
            for (var idx_oc = 0u; idx_oc < output_channels; idx_oc = idx_oc + 1u) {
                // output = output + 
                //     get_src0(idx_oc, global_id.y, ik) * 
                //     get_src1(mult_idx - idx_offset, idx_oc, global_id.z);
                let val_src0 = f32(get_src0_lin(base_idx_src0 + idx_oc));
                let idx_src1 = (base_idx_src1 + idx_oc * tensor_dimension_params.src[1].nb[1]) / 4u;
                output = output + val_src0 * mult1 * vec4f(src1_v4[idx_src1]);
            }
        }
    }

    let dst_idx = (global_id.z * tensor_dimension_params.dst.nb[2] +
        global_id.y * tensor_dimension_params.dst.nb[1] +
        mult_idx) / 4u;
    dst_v4[dst_idx] = vec4h(output);
}

);


static const char src_ggml_shader_kernel_acc[] = MULTILINE(

@compute
@workgroup_size(256)
fn kernel_acc(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let nb0 = 4u;
    let nb1 = u32(tensor_dimension_params.params[0][0]);
    let nb2 = u32(tensor_dimension_params.params[0][1]);
    let nb3 = u32(tensor_dimension_params.params[0][2]);
    let offset = u32(tensor_dimension_params.params[0][3]);
    let inplace = bool(tensor_dimension_params.params[1][0]);
    let zero_out_accumulator = bool(tensor_dimension_params.params[1][1]);

    let nc = u32(tensor_dimension_params.src[1].ne[0]);
    let nc_out = u32(tensor_dimension_params.dst.ne[0]);
    let offset_ne = offset / 2u;
    let nc_limit = nc + offset_ne;
    let nc_after = nc_out - nc_limit;

    if (global_id.x >= nc_out) {
        return;
    }

    var output : f32 = 0.0;
    if (!zero_out_accumulator) {
        output = f32(get_src0(global_id.x, global_id.y, global_id.z));
    }

    // var dst_addr_offset = global_id.z * tensor_dimension_params.dst.nb[2] +
    //     global_id.y * tensor_dimension_params.dst.nb[1] +
    //     global_id.x * tensor_dimension_params.dst.nb[0];

    // if (dst_addr_offset >= offset) {
    //     dst_addr_offset = dst_addr_offset - offset;
    //     let idx2 = dst_addr_offset / nb2;
    //     dst_addr_offset = dst_addr_offset - idx2 * nb2;
    //     let idx1 = dst_addr_offset / nb1;
    //     dst_addr_offset = dst_addr_offset - idx1 * nb1;
    //     let idx0 = dst_addr_offset / nb0;
    //     if (idx0 < nc && idx1 < u32(tensor_dimension_params.src[1].ne[1]) && idx2 < u32(tensor_dimension_params.src[1].ne[2])) {
    //         output = output + get_src1(idx0, idx1, idx2);
    //     }
    // }

    if (global_id.x >= offset_ne && global_id.x < nc_limit) {
        let idx0 = global_id.x - offset_ne;
        output = output + f32(get_src1(idx0, global_id.y, global_id.z));
    }

    set_dst(global_id.x, global_id.y, global_id.z, f16(output));
}

);


static const char src_ggml_shader_kernel_add_and_tanh_back[] = MULTILINE(

@compute
@workgroup_size(256)
fn kernel_add_and_tanh_back(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let mult_idx = global_id.x * 4u;

    if (mult_idx >= num_el_dst()) {
        return;
    }

    let x = vec4f(src0_v4[global_id.x]);
    let y = vec4f(src1_v4[global_id.x]);
    let z = (1.0 - x*x)*y;

    dst_v4[global_id.x] = vec4h(z);
}

);


static const char src_ggml_shader_kernel_add[] = MULTILINE(

@compute
@workgroup_size(256)
fn kernel_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(tensor_dimension_params.dst.ne[0])) {
        return;
    }

    let idx0 = global_id.x * u32(tensor_dimension_params.src[1].ne[0]) / u32(tensor_dimension_params.dst.ne[0]);
    let idx1 = global_id.y * u32(tensor_dimension_params.src[1].ne[1]) / u32(tensor_dimension_params.dst.ne[1]);
    let idx2 = global_id.z * u32(tensor_dimension_params.src[1].ne[2]) / u32(tensor_dimension_params.dst.ne[2]);

    set_dst(global_id.x, global_id.y, global_id.z, 
        get_src0(global_id.x, global_id.y, global_id.z) + get_src1(idx0, idx1, idx2));
}

);


static const char src_ggml_shader_kernel_conv_1d_small_kern_back_bias[] = MULTILINE(

var<workgroup> workgroup_data: array<f32, 256>;

@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern_back_bias(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let output_channels = u32(tensor_dimension_params.dst.ne[1]);
    let output_len = u32(tensor_dimension_params.src[0].ne[0]);
    let num_batches = u32(tensor_dimension_params.src[0].ne[2]);

    var output : f32 = 0.0;

    for (var ir = 0u; ir < num_batches; ir = ir + 1u) {
        let base_idx_src0_base = wg_id.x * tensor_dimension_params.src[0].nb[1] + ir * tensor_dimension_params.src[0].nb[2];
        // TODO: handle output_len not being a multiple of 4
        for (var isample = 4u*local_id.x; isample < output_len; isample = isample + 4u*256u) {
            let base_idx_src0 = base_idx_src0_base + isample;
            output = output + f32(get_src0_lin(base_idx_src0));
            output = output + f32(get_src0_lin(base_idx_src0+1u));
            output = output + f32(get_src0_lin(base_idx_src0+2u));
            output = output + f32(get_src0_lin(base_idx_src0+3u));
        }
    }

    workgroup_data[local_id.x] = output;
    workgroupBarrier();

    if (0u == local_id.x) {
        output = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            output = output + workgroup_data[i];
        }

        set_dst(0u, wg_id.x, 0u, f16(output));
    }

}


);

static const char src_ggml_shader_kernel_conv_1d_small_kern_back_bias_stage1[] = MULTILINE(

@group(0) @binding(5)
var<storage,read_write> src5_f32: array<f32>;

var<workgroup> workgroup_data_v4f: array<vec4f, 256>;

@compute
@workgroup_size(256)
fn kernel_conv_1d_small_kern_back_bias_stage1(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let output_len = u32(tensor_dimension_params.src[0].ne[0]);
    let num_batches = u32(tensor_dimension_params.src[0].ne[2]);

    var output = vec4f();

    let base_idx_src0_base = (wg_id.x * tensor_dimension_params.src[0].nb[1] + wg_id.y * tensor_dimension_params.src[0].nb[2]) / 4u;
    for (var isample = local_id.x; isample < ((output_len+3u)/4u); isample = isample + 256u) {
        let mult1 = vec4f(
            f32((4u * isample) < output_len),
            f32((4u * isample + 1u) < output_len),
            f32((4u * isample + 2u) < output_len),
            f32((4u * isample + 3u) < output_len)
        );

        output = output + mult1 * vec4f(src0_v4[base_idx_src0_base + isample]);
    }

    workgroup_data_v4f[local_id.x] = output;
    workgroupBarrier();

    if (0u == local_id.x) {
        output = vec4f();
        for (var i = 0u; i < 256u; i = i + 1u) {
            output = output + workgroup_data_v4f[i];
        }

        src5_f32[wg_id.y + wg_id.x * num_batches] = output.x+output.y+output.z+output.w;
        // set_src5_lin(wg_id.y + wg_id.x * num_batches, f16(output.x+output.y+output.z+output.w));
    }
}

);


static const char src_ggml_shader_kernel_conv_1d_small_kern_back_bias_stage2[] = MULTILINE(

@group(0) @binding(5)
var<storage,read_write> src5_f32: array<f32>;


@compute
@workgroup_size(1)
fn kernel_conv_1d_small_kern_back_bias_stage2(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let num_batches = u32(tensor_dimension_params.src[0].ne[2]);

    var output : f32 = 0.0;

    let base_idx_src1 = wg_id.x * num_batches;
    for (var isample = 0u; isample < num_batches; isample = isample + 1u) {
        output = output + src5_f32[base_idx_src1 + isample];
    }
    set_dst(0u, wg_id.x, 0u, f16(output));
}

);


static const char src_ggml_shader_kernel_special_adam_step[] = MULTILINE(

@group(0) @binding(2)
var<storage,read_write> src2_f32: array<f32>;

@group(0) @binding(3)
var<storage,read_write> src3_f32: array<f32>;

@group(0) @binding(4)
var<storage,read_write> src4_f32: array<f32>;

@compute
@workgroup_size(256)
fn kernel_special_adam_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_el_dst()) {
        return;
    }

    let beta1 = bitcast<f32>(tensor_dimension_params.params[0][0]);
    let beta2 = bitcast<f32>(tensor_dimension_params.params[0][1]);
    let beta1h = bitcast<f32>(tensor_dimension_params.params[0][2]);
    let beta2h = bitcast<f32>(tensor_dimension_params.params[0][3]);
    let eps = bitcast<f32>(tensor_dimension_params.params[1][0]);
    let gradient_scale = bitcast<f32>(tensor_dimension_params.params[1][1]);

    var x = src4_f32[global_id.x];
    var g = f32(get_src1_lin(global_id.x)) * gradient_scale;
    var m = src2_f32[global_id.x];
    var v = src3_f32[global_id.x];

    m = m*beta1 +   g*(1.0 - beta1);
    v = v*beta2 + g*g*(1.0 - beta2);
    let mh = m*beta1h;
    let vh = sqrt(v*beta2h) + eps;
    x = x - mh/vh;

    src2_f32[global_id.x] = m;
    src3_f32[global_id.x] = v;
    src4_f32[global_id.x] = x;
    set_dst_lin(global_id.x, f16(x));
}

);


static const char src_ggml_shader_kernel_special_adam_step_inplace[] = MULTILINE(

@group(0) @binding(2)
var<storage,read_write> src2_f32: array<f32>;

@group(0) @binding(3)
var<storage,read_write> src3_f32: array<f32>;

@group(0) @binding(4)
var<storage,read_write> src4_f32: array<f32>;

@compute
@workgroup_size(256)
fn kernel_special_adam_step_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_el_dst()) {
        return;
    }

    let beta1 = bitcast<f32>(tensor_dimension_params.params[0][0]);
    let beta2 = bitcast<f32>(tensor_dimension_params.params[0][1]);
    let beta1h = bitcast<f32>(tensor_dimension_params.params[0][2]);
    let beta2h = bitcast<f32>(tensor_dimension_params.params[0][3]);
    let eps = bitcast<f32>(tensor_dimension_params.params[1][0]);
    let gradient_scale = bitcast<f32>(tensor_dimension_params.params[1][1]);

    var x = src4_f32[global_id.x];
    var g = f32(get_src1_lin(global_id.x)) * gradient_scale;
    var m = src2_f32[global_id.x];
    var v = src3_f32[global_id.x];

    m = m*beta1 +   g*(1.0 - beta1);
    v = v*beta2 + g*g*(1.0 - beta2);
    let mh = m*beta1h;
    let vh = sqrt(v*beta2h) + eps;
    x = x - mh/vh;

    src2_f32[global_id.x] = m;
    src3_f32[global_id.x] = v;
    src4_f32[global_id.x] = x;
    set_dst_lin(global_id.x, f16(x));
}

);

#undef MULTILINE



#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef GGML_WGPU_NDEBUG
#define GGML_WGPU_LOG_INFO(...)
#define GGML_WGPU_LOG_WARN(...)
#define GGML_WGPU_LOG_ERROR(...)
#else
#define GGML_WGPU_LOG_INFO(...)  ggml_wgpu_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_WGPU_LOG_WARN(...)  ggml_wgpu_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_WGPU_LOG_ERROR(...) ggml_wgpu_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#endif


struct ggml_wgpu_buffer {
    const char * name;

    void   * data;
    size_t   size;

    WGPUBuffer wgpu;
};


struct ggml_wgpu_dim_param {
    int32_t ne[4];
    uint32_t nb[4];
    uint32_t offset;
    uint32_t ph1;
    uint32_t ph2;
    uint32_t ph3;
};

struct ggml_wgpu_operator_params {
    struct ggml_wgpu_dim_param tensor_dimension_params[GGML_WGPU_DIM_PARAMS_SIZE];
    int32_t op_params[GGML_WGPU_OP_PARAMS_SIZE];
};

const size_t GGML_WGPU_OPERATOR_PARAMS_SIZE_TOTAL = CEIL_DIV(sizeof(struct ggml_wgpu_operator_params) , MIN_STORAGE_BUFFER_ALIGNMENT) * MIN_STORAGE_BUFFER_ALIGNMENT;

struct ggml_wgpu_context {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUSupportedLimits limits;
    WGPUQueue queue;
    WGPUBindGroupLayout bind_group_layout;
    WGPUPipelineLayout pipeline_layout;

    WGPUBuffer tensor_dimension_operation_params;
    WGPUBuffer placeholder_buffer[GGML_MAX_SRC];
    WGPUBuffer placeholder_uniform_buffer[GGML_WGPU_NUM_EXTRA_UNIFORM_BINDINGS];
    struct ggml_wgpu_operator_params tensor_dimension_operation_params_host[GGML_MAX_NODES];

    WGPUBindGroupEntry bind_group_entries[GGML_WGPU_BINDINGS_SIZE];

    int n_buffers;
    struct ggml_wgpu_buffer buffers[GGML_WGPU_MAX_BUFFERS];
    bool wgpu_buffer_mapped_flag;

    WGPUQuerySet timestamp_queries;
    WGPUBuffer timestamp_queries_resolve_buffer;
    WGPUBuffer timestamp_queries_read_buffer;

    // custom kernels
#define GGML_WGPU_DECL_KERNEL(name) \
    WGPUShaderModule shader_module_##name; \
    WGPUComputePipeline pipeline_##name;

    GGML_WGPU_DECL_KERNEL(silu)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_simpl)
    // GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_opti)
    // GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_opti_large_dil)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_no_offsets)
    // GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_no_offsets_8x8x3)
    GGML_WGPU_DECL_KERNEL(add_and_trim)
    GGML_WGPU_DECL_KERNEL(scale)
    GGML_WGPU_DECL_KERNEL(scale_inplace)
    GGML_WGPU_DECL_KERNEL(sub)
    GGML_WGPU_DECL_KERNEL(sqr)
    GGML_WGPU_DECL_KERNEL(sum)
    GGML_WGPU_DECL_KERNEL(repeat)
    GGML_WGPU_DECL_KERNEL(mul)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_back_filter)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_back_input)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_back_input_large_dil)
    GGML_WGPU_DECL_KERNEL(acc)
    GGML_WGPU_DECL_KERNEL(add_and_tanh_back)
    GGML_WGPU_DECL_KERNEL(add)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_back_bias)
    GGML_WGPU_DECL_KERNEL(special_adam_step)
    GGML_WGPU_DECL_KERNEL(special_adam_step_inplace)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_back_bias_stage1)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_back_bias_stage2)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_back_filter_stage1)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_back_filter_stage2)
    GGML_WGPU_DECL_KERNEL(conv_1d_small_kern_back_filter_nk1)

#undef GGML_WGPU_DECL_KERNEL
};


static void wait_for_buffer_map(struct ggml_wgpu_context* ctx) {
    while (!(ctx->wgpu_buffer_mapped_flag)) {
#ifdef WEBGPU_BACKEND_WGPU
        // Non-standardized behavior: submit empty queue to flush callbacks
        // (wgpu-native also has a wgpuDevicePoll but its API is more complex)
        // wgpuQueueSubmit(ctx->queue, 0, NULL);
        wgpuDevicePoll(ctx->device, true, NULL);
#else
        // Non-standard Dawn way
        wgpuDeviceTick(ctx->device);
        sleep_ms(1);
#endif
    }
    ctx->wgpu_buffer_mapped_flag = false;
}
static void handle_buffer_map(WGPUBufferMapAsyncStatus status, void *userdata) {
    struct ggml_wgpu_context* ctx = userdata;
    if(status != WGPUBufferMapAsyncStatus_Success) {
        printf(LOG_PREFIX " buffer_map status=%#.8x\n", status);
    } else {
        ctx->wgpu_buffer_mapped_flag = true;
    }
}


ggml_log_callback ggml_wgpu_log_callback = NULL;
void * ggml_wgpu_log_user_data = NULL;

void ggml_wgpu_log_set_callback(ggml_log_callback log_callback, void * user_data) {
    ggml_wgpu_log_callback  = log_callback;
    ggml_wgpu_log_user_data = user_data;
}

#ifdef WEBGPU_BACKEND_WGPU
static void wgpu_log_callback(WGPULogLevel level, char const *message,
                         void *userdata) {
  UNUSED(userdata);
  char *level_str;
  switch (level) {
  case WGPULogLevel_Error:
    level_str = "error";
    break;
  case WGPULogLevel_Warn:
    level_str = "warn";
    break;
  case WGPULogLevel_Info:
    level_str = "info";
    break;
  case WGPULogLevel_Debug:
    level_str = "debug";
    break;
  case WGPULogLevel_Trace:
    level_str = "trace";
    break;
  default:
    level_str = "unknown_level";
  }
  fprintf(stderr, "[wgpu] [%s] %s\n", level_str, message);
}
#endif

static void ggml_wgpu_log(enum ggml_log_level level, const char* format, ...){
    if (ggml_wgpu_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_wgpu_log_callback(level, buffer, ggml_wgpu_log_user_data);
        } else {
            char* buffer2 = malloc(len+1);
            vsnprintf(buffer2, len+1, format, args);
            buffer2[len] = 0;
            ggml_wgpu_log_callback(level, buffer2, ggml_wgpu_log_user_data);
            free(buffer2);
        }
        va_end(args);
    }
}

static void new_er_log(WGPUErrorType type, char const* message, void* userdata)
{
    printf("[new_er_log] Error type %u: %s\n", type, message);
    GGML_ASSERT(false);
}


struct ggml_wgpu_context * ggml_wgpu_init(void) {
    GGML_WGPU_LOG_INFO("%s: allocating\n", __func__);

#ifdef WEBGPU_BACKEND_WGPU
    wgpuSetLogCallback(wgpu_log_callback, NULL);
    wgpuSetLogLevel(WGPULogLevel_Info);
#endif


    // Configure context
    struct ggml_wgpu_context * ctx = malloc(sizeof(struct ggml_wgpu_context));
    memset(ctx, 0, sizeof(struct ggml_wgpu_context));

    ctx->wgpu_buffer_mapped_flag = false;

    WGPUInstanceDescriptor desc = {0};

    ctx->instance = wgpuCreateInstance(&desc);
    ASSERT_CHECK(ctx->instance);

    WGPURequestAdapterOptions reqAdOptions = {0};
    reqAdOptions.powerPreference = WGPUPowerPreference_HighPerformance;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    // reqAdOptions.backendType = WGPUBackendType_D3D12;
    reqAdOptions.backendType = WGPUBackendType_Vulkan;
#elif __APPLE__
    reqAdOptions.backendType = WGPUBackendType_Metal;
#elif __linux__
    reqAdOptions.backendType = WGPUBackendType_Vulkan;
#endif
    wgpuInstanceRequestAdapter(ctx->instance, &reqAdOptions, handle_request_adapter,
                                (void *)&(ctx->adapter));
    ASSERT_CHECK(ctx->adapter);

    WGPUDeviceDescriptor deviceDesc = {0};
    WGPUFeatureName features[2];
#if GGML_PERF
    if (wgpuAdapterHasFeature(ctx->adapter, WGPUFeatureName_TimestampQuery)) {
        features[deviceDesc.requiredFeatureCount++] = WGPUFeatureName_TimestampQuery;
    }
#endif
#if WEBGPU_BACKEND_DAWN
    if (wgpuAdapterHasFeature(ctx->adapter, WGPUFeatureName_ShaderF16)) {
        features[deviceDesc.requiredFeatureCount++] = WGPUFeatureName_ShaderF16;
    }
#endif
    
    deviceDesc.requiredFeatures = features;
    wgpuAdapterRequestDevice(ctx->adapter, &deviceDesc, handle_request_device,
                            (void *)&(ctx->device));
    ASSERT_CHECK(ctx->device);

    wgpuDeviceSetUncapturedErrorCallback(ctx->device, new_er_log, NULL);


#if GGML_PERF
    if (wgpuDeviceHasFeature(ctx->device, WGPUFeatureName_TimestampQuery)) {
        GGML_WGPU_LOG_INFO("Enabling timestamp queries\n");
        // Create timestamp queries
        WGPUQuerySetDescriptor querySetDesc = {0};
        querySetDesc.nextInChain = NULL;
        querySetDesc.type = WGPUQueryType_Timestamp;
        querySetDesc.count = GGML_MAX_NODES+1;
        ctx->timestamp_queries = wgpuDeviceCreateQuerySet(ctx->device, &querySetDesc);
        ASSERT_CHECK(ctx->timestamp_queries);

        ctx->timestamp_queries_resolve_buffer = wgpuDeviceCreateBuffer(ctx->device, &(const WGPUBufferDescriptor){
                                                                .label = "timestamp_queries_resolve_buffer",
                                                                .usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc,
                                                                .size = 8*querySetDesc.count,
                                                                .mappedAtCreation = false,
                                                            });
        ASSERT_CHECK(ctx->timestamp_queries_resolve_buffer);

        ctx->timestamp_queries_read_buffer = wgpuDeviceCreateBuffer(ctx->device, &(const WGPUBufferDescriptor){
                                                                .label = "timestamp_queries_read_buffer",
                                                                .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
                                                                .size = 8*querySetDesc.count,
                                                                .mappedAtCreation = false,
                                                            });
        ASSERT_CHECK(ctx->timestamp_queries_read_buffer);
    } else {
        GGML_WGPU_LOG_INFO("Timestamp queries not supported\n");
    }
#endif

    ASSERT_CHECK(wgpuDeviceGetLimits(ctx->device, &(ctx->limits)));

    ctx->queue = wgpuDeviceGetQueue(ctx->device);
    ASSERT_CHECK(ctx->queue);

    ctx->tensor_dimension_operation_params = wgpuDeviceCreateBuffer(ctx->device, &(const WGPUBufferDescriptor){
                                                            .label = "tensor_dimension_operation_params",
                                                            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                                                            .size = GGML_WGPU_OPERATOR_PARAMS_SIZE_TOTAL * GGML_MAX_NODES,
                                                            .mappedAtCreation = false,
                                                         });
    ASSERT_CHECK(ctx->tensor_dimension_operation_params);

    for (int i=0; i<GGML_MAX_SRC; i++) {
        char placeholder_buffer_name[30];
        sprintf(placeholder_buffer_name, "placeholder_buffer_%d", i);
        ctx->placeholder_buffer[i] = wgpuDeviceCreateBuffer(ctx->device, &(const WGPUBufferDescriptor){
                                                                .label = placeholder_buffer_name,
                                                                .usage = WGPUBufferUsage_Storage,
                                                                .size = PLACEHOLDER_BUFFER_SIZE, // making it slightly bigger to use as scratch space when needed
                                                                .mappedAtCreation = false,
                                                            });
        ASSERT_CHECK(ctx->placeholder_buffer[i]);
    }

    for (int i=0; i<GGML_WGPU_NUM_EXTRA_UNIFORM_BINDINGS; i++) {
        char placeholder_buffer_name[36];
        sprintf(placeholder_buffer_name, "placeholder_uniform_buffer_%d", i);
        ctx->placeholder_uniform_buffer[i] = wgpuDeviceCreateBuffer(ctx->device, &(const WGPUBufferDescriptor){
                                                                    .label = placeholder_buffer_name,
                                                                    .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
                                                                    .size = GGML_WGPU_EXTRA_UNIFORM_SIZE * GGML_MAX_NODES,
                                                                    .mappedAtCreation = false,
                                                                });
        ASSERT_CHECK(ctx->placeholder_uniform_buffer[i]);
    }

    ctx->bind_group_entries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].binding = GGML_WGPU_DIM_PARAMS_BINDING_INDEX;
    ctx->bind_group_entries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].buffer = ctx->tensor_dimension_operation_params;
    ctx->bind_group_entries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].offset = 0;
    ctx->bind_group_entries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].size = sizeof(struct ggml_wgpu_operator_params);



    ctx->n_buffers = 0;

    WGPUBindGroupLayoutEntry bindGroupLayoutEntries[GGML_WGPU_BINDINGS_SIZE] = {0};
    {
        for (int i = 0; i < GGML_WGPU_DIM_PARAMS_BINDING_INDEX; ++i) {
            bindGroupLayoutEntries[i].binding = i;
            bindGroupLayoutEntries[i].visibility = WGPUShaderStage_Compute;
            bindGroupLayoutEntries[i].buffer.type = WGPUBufferBindingType_Storage;
            bindGroupLayoutEntries[i].buffer.hasDynamicOffset = false;
            bindGroupLayoutEntries[i].buffer.minBindingSize = 0;
        }

        bindGroupLayoutEntries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].binding = GGML_WGPU_DIM_PARAMS_BINDING_INDEX;
        bindGroupLayoutEntries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].visibility = WGPUShaderStage_Compute;
        bindGroupLayoutEntries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].buffer.type = WGPUBufferBindingType_Uniform;
        bindGroupLayoutEntries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].buffer.hasDynamicOffset = false;
        bindGroupLayoutEntries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].buffer.minBindingSize = sizeof(struct ggml_wgpu_operator_params);

        for (int i = 0; i < GGML_WGPU_NUM_EXTRA_UNIFORM_BINDINGS; ++i) {
            bindGroupLayoutEntries[GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + i].binding = GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + i;
            bindGroupLayoutEntries[GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + i].visibility = WGPUShaderStage_Compute;
            bindGroupLayoutEntries[GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + i].buffer.type = WGPUBufferBindingType_Uniform;
            bindGroupLayoutEntries[GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + i].buffer.hasDynamicOffset = false;
            bindGroupLayoutEntries[GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + i].buffer.minBindingSize = GGML_WGPU_EXTRA_UNIFORM_SIZE;
        }
    }


    ctx->bind_group_layout = wgpuDeviceCreateBindGroupLayout(ctx->device, &(const WGPUBindGroupLayoutDescriptor){
                           .label = "ggml-wgpu-bind-group-layout",
                           .entries = bindGroupLayoutEntries,
                           .entryCount = GGML_WGPU_BINDINGS_SIZE,
                       });
    ASSERT_CHECK(ctx->bind_group_layout);

    ctx->pipeline_layout = wgpuDeviceCreatePipelineLayout(ctx->device, &(const WGPUPipelineLayoutDescriptor){
                           .label = "ggml-wgpu-pipeline-layout",
                           .bindGroupLayoutCount = 1,
                           .bindGroupLayouts = &(ctx->bind_group_layout),
                       });
    ASSERT_CHECK(ctx->pipeline_layout);




    // load kernels
    {
#define GGML_WGPU_ADD_KERNEL(name) \
        { \
            char * shader_src = malloc(sizeof(src_ggml_shader_common_0) + sizeof(src_ggml_shader_common_1) + sizeof(src_ggml_shader_common_2) + sizeof(src_ggml_shader_kernel_##name) + 1); \
            memset(shader_src, 0, sizeof(src_ggml_shader_common_0) + sizeof(src_ggml_shader_common_1) + sizeof(src_ggml_shader_common_2) + sizeof(src_ggml_shader_kernel_##name) + 1); \
            strcat(shader_src, src_ggml_shader_common_0);               \
            strcat(shader_src, src_ggml_shader_common_1);               \
            strcat(shader_src, src_ggml_shader_common_2);               \
            strcat(shader_src, src_ggml_shader_kernel_##name);          \
            ctx->shader_module_##name = wgpuDeviceCreateShaderModule(   \
            ctx->device, &(const WGPUShaderModuleDescriptor){           \
                        .label = "ggml_shader_module_" #name,           \
                        .nextInChain =                                  \
                            (const WGPUChainedStruct *)&(               \
                                const WGPUShaderModuleWGSLDescriptor){  \
                                .chain =                                \
                                    (const WGPUChainedStruct){          \
                                        .sType = WGPUSType_ShaderModuleWGSLDescriptor, \
                                    },                                  \
                                .code = shader_src,                     \
                            },                                          \
                    });                                                 \
            ASSERT_CHECK(ctx->shader_module_##name);                    \
            ctx->pipeline_##name = wgpuDeviceCreateComputePipeline(     \
                ctx->device, &(const WGPUComputePipelineDescriptor){    \
                            .label = "compute_pipeline_" #name,         \
                            .layout = ctx->pipeline_layout,             \
                            .compute =                                  \
                                (const WGPUProgrammableStageDescriptor){\
                                    .module = ctx->shader_module_##name,\
                                    .entryPoint = "kernel_" #name,      \
                                },                                      \
                        });                                             \
            ASSERT_CHECK(ctx->pipeline_##name);                         \
            free(shader_src); \
        }


        GGML_WGPU_ADD_KERNEL(silu);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_simpl);
        // GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_opti);
        // GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_opti_large_dil);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_no_offsets);
        // GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_no_offsets_8x8x3);
        GGML_WGPU_ADD_KERNEL(add_and_trim);
        GGML_WGPU_ADD_KERNEL(scale);
        GGML_WGPU_ADD_KERNEL(scale_inplace);
        GGML_WGPU_ADD_KERNEL(sub);
        GGML_WGPU_ADD_KERNEL(sqr);
        GGML_WGPU_ADD_KERNEL(sum);
        GGML_WGPU_ADD_KERNEL(repeat);
        GGML_WGPU_ADD_KERNEL(mul);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_back_filter);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_back_input);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_back_input_large_dil);
        GGML_WGPU_ADD_KERNEL(acc);
        GGML_WGPU_ADD_KERNEL(add_and_tanh_back);
        GGML_WGPU_ADD_KERNEL(add);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_back_bias);
        GGML_WGPU_ADD_KERNEL(special_adam_step);
        GGML_WGPU_ADD_KERNEL(special_adam_step_inplace);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_back_bias_stage1);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_back_bias_stage2);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_back_filter_stage1);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_back_filter_stage2);
        GGML_WGPU_ADD_KERNEL(conv_1d_small_kern_back_filter_nk1);

#undef GGML_WGPU_ADD_KERNEL
    }

    return ctx;
}

void ggml_wgpu_free(struct ggml_wgpu_context * ctx) {
    GGML_WGPU_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < ctx->n_buffers; ++i) {
        if (ctx->buffers[i].wgpu) wgpuBufferRelease(ctx->buffers[i].wgpu);
    }

#define GGML_WGPU_DEL_KERNEL(name) \
    if (ctx->pipeline_##name) wgpuComputePipelineRelease(ctx->pipeline_##name); \
    if (ctx->shader_module_##name) wgpuShaderModuleRelease(ctx->shader_module_##name);

    GGML_WGPU_DEL_KERNEL(silu)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_simpl)
    // GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_opti)
    // GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_opti_large_dil)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_no_offsets)
    // GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_no_offsets_8x8x3)
    GGML_WGPU_DEL_KERNEL(add_and_trim)
    GGML_WGPU_DEL_KERNEL(scale)
    GGML_WGPU_DEL_KERNEL(scale_inplace)
    GGML_WGPU_DEL_KERNEL(sub)
    GGML_WGPU_DEL_KERNEL(sqr)
    GGML_WGPU_DEL_KERNEL(sum)
    GGML_WGPU_DEL_KERNEL(repeat)
    GGML_WGPU_DEL_KERNEL(mul)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_back_filter)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_back_input)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_back_input_large_dil)
    GGML_WGPU_DEL_KERNEL(acc)
    GGML_WGPU_DEL_KERNEL(add_and_tanh_back)
    GGML_WGPU_DEL_KERNEL(add)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_back_bias)
    GGML_WGPU_DEL_KERNEL(special_adam_step)
    GGML_WGPU_DEL_KERNEL(special_adam_step_inplace)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_back_bias_stage1)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_back_bias_stage2)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_back_filter_stage1)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_back_filter_stage2)
    GGML_WGPU_DEL_KERNEL(conv_1d_small_kern_back_filter_nk1)

#undef GGML_WGPU_DEL_KERNEL
    

    if (ctx->pipeline_layout) wgpuPipelineLayoutRelease(ctx->pipeline_layout);
    if (ctx->bind_group_layout) wgpuBindGroupLayoutRelease(ctx->bind_group_layout);
    for (int i=0; i<GGML_WGPU_NUM_EXTRA_UNIFORM_BINDINGS; i++) {
        if (ctx->placeholder_uniform_buffer[i]) wgpuBufferRelease(ctx->placeholder_uniform_buffer[i]);
    }
    for (int i=0; i<GGML_MAX_SRC; i++) {
        if (ctx->placeholder_buffer[i]) wgpuBufferRelease(ctx->placeholder_buffer[i]);
    }
    if (ctx->tensor_dimension_operation_params) wgpuBufferRelease(ctx->tensor_dimension_operation_params);
    if (ctx->queue) wgpuQueueRelease(ctx->queue);
    if (ctx->timestamp_queries_resolve_buffer) wgpuBufferRelease(ctx->timestamp_queries_resolve_buffer);
    if (ctx->timestamp_queries_read_buffer) wgpuBufferRelease(ctx->timestamp_queries_read_buffer);
    if (ctx->timestamp_queries) wgpuQuerySetRelease(ctx->timestamp_queries);
    if (ctx->device) wgpuDeviceRelease(ctx->device);
    if (ctx->adapter) wgpuAdapterRelease(ctx->adapter);
    if (ctx->instance) wgpuInstanceRelease(ctx->instance);
    free(ctx);
}


void * ggml_wgpu_host_malloc(size_t n) {
    void * data = NULL;
#if _MSC_VER
    GGML_ASSERT(false); // not implemented
#else
    const int result = posix_memalign((void **) &data, MIN_STORAGE_BUFFER_ALIGNMENT, n);
    if (result != 0) {
        GGML_WGPU_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif
    return data;
}

void ggml_wgpu_host_free(void * data) {
    free(data);
}

// finds the WebGPU buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// WebGPU buffer based on the host memory pointer
//
static WGPUBuffer ggml_wgpu_get_buffer(struct ggml_wgpu_context * ctx, struct ggml_tensor * t, size_t * offs) {
    //GGML_WGPU_LOG_INFO("%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) ctx->buffers[i].data;

        //GGML_WGPU_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, ctx->buffers[%d].size = %10ld, name = %s\n", ioffs, tsize, ioffs + tsize, i, ctx->buffers[i].size, ctx->buffers[i].name);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            //GGML_WGPU_LOG_INFO("%s: '%s' tensor '%16s', offs = %8ld\n", __func__, ctx->buffers[i].name, t->name, *offs);

            return ctx->buffers[i].wgpu;
        }
    }

    GGML_WGPU_LOG_ERROR("%s: error: buffer is null\n", __func__);

    return NULL;
}

bool ggml_wgpu_add_buffer(
        struct ggml_wgpu_context * ctx,
                     const char * name,
                           void * data,
                         size_t   size,
                         size_t   max_size) {
    if (ctx->n_buffers >= GGML_WGPU_MAX_BUFFERS) {
        GGML_WGPU_LOG_ERROR("%s: error: too many buffers\n", __func__);
        return false;
    }

    if (data) {
        // verify that the buffer does not overlap with any of the existing buffers
        for (int i = 0; i < ctx->n_buffers; ++i) {
            const int64_t ioffs = (int64_t) data - (int64_t) ctx->buffers[i].data;

            if (ioffs >= 0 && ioffs < (int64_t) ctx->buffers[i].size) {
                GGML_WGPU_LOG_ERROR("%s: error: buffer '%s' overlaps with '%s'\n", __func__, name, ctx->buffers[i].name);
                return false;
            }
        }

        const size_t size_page = MIN_STORAGE_BUFFER_ALIGNMENT; // TODO: figure out if this needs a real value like on metal // sysconf(_SC_PAGESIZE);

        size_t size_aligned = size;
        if ((size_aligned % size_page) != 0) {
            size_aligned += (size_page - (size_aligned % size_page));
        }

        // the buffer fits into the max buffer size allowed by the device
        if (size_aligned <= ctx->limits.limits.maxBufferSize) {
            ctx->buffers[ctx->n_buffers].name = name;
            ctx->buffers[ctx->n_buffers].data = data;
            ctx->buffers[ctx->n_buffers].size = size;

            // TODO: proper buffer label
            ctx->buffers[ctx->n_buffers].wgpu = wgpuDeviceCreateBuffer(
                                                    ctx->device, &(const WGPUBufferDescriptor){
                                                                .label = "storage_buffer",
                                                                .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                                                                        WGPUBufferUsage_CopySrc,
                                                                .size = size_aligned,
                                                                .mappedAtCreation = false,
                                                });

            if (ctx->buffers[ctx->n_buffers].wgpu == NULL) {
                GGML_WGPU_LOG_ERROR("%s: error: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_aligned / 1024.0 / 1024.0);
                return false;
            }
            wgpuQueueWriteBuffer(ctx->queue, ctx->buffers[ctx->n_buffers].wgpu, 0, data, size);

            GGML_WGPU_LOG_INFO("%s: allocated '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_aligned / 1024.0 / 1024.0);

            ++ctx->n_buffers;
        } else {
            // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
            // one of the views
            const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
            const size_t size_step = ctx->limits.limits.maxBufferSize - size_ovlp;
            const size_t size_view = ctx->limits.limits.maxBufferSize;

            for (size_t i = 0; i < size; i += size_step) {
                const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

                ctx->buffers[ctx->n_buffers].name = name;
                ctx->buffers[ctx->n_buffers].data = (void *) ((uint8_t *) data + i);
                ctx->buffers[ctx->n_buffers].size = size_step_aligned;

                // TODO: proper buffer label
                ctx->buffers[ctx->n_buffers].wgpu = wgpuDeviceCreateBuffer(
                                                        ctx->device, &(const WGPUBufferDescriptor){
                                                                    .label = "storage_buffer",
                                                                    .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                                                                            WGPUBufferUsage_CopySrc,
                                                                    .size = size_step_aligned,
                                                                    .mappedAtCreation = false,
                                                    });



                if (ctx->buffers[ctx->n_buffers].wgpu == NULL) {
                    GGML_WGPU_LOG_ERROR("%s: error: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_step_aligned / 1024.0 / 1024.0);
                    return false;
                }
                // TODO: might be copying bytes out of range if the original alignment is not at 4bytes
                wgpuQueueWriteBuffer(ctx->queue, ctx->buffers[ctx->n_buffers].wgpu, 0, (void *) ((uint8_t *) data + i), size_step_aligned);

                GGML_WGPU_LOG_INFO("%s: allocated '%-16s' buffer, size = %8.2f MB, offs = %12ld\n", __func__, name, size_step_aligned / 1024.0 / 1024.0, i);
                if (i + size_step < size) {
                    GGML_WGPU_LOG_INFO("\n");
                }

                ++ctx->n_buffers;
            }
        }

        // GGML_WGPU_LOG_INFO(", (%8.2f)\n", ctx->device.currentAllocatedSize / 1024.0 / 1024.0);
    }

    return true;
}


void ggml_wgpu_read_back_buffer(
        struct ggml_wgpu_context * ctx,
                       const char * name) {
    struct ggml_wgpu_buffer * buffer = NULL;
    for (int i = 0; i < ctx->n_buffers; ++i) {
        if (strcmp(ctx->buffers[i].name, name) == 0) {
            GGML_ASSERT(buffer == NULL); // this function does not yet support reading split buffers
            buffer = &ctx->buffers[i];
        }
    }

    GGML_ASSERT(buffer);

    const size_t original_size = buffer->size;

    size_t size_aligned = original_size;
    if ((size_aligned % MIN_STORAGE_BUFFER_ALIGNMENT) != 0) {
        size_aligned += (MIN_STORAGE_BUFFER_ALIGNMENT - (size_aligned % MIN_STORAGE_BUFFER_ALIGNMENT));
    }

    WGPUBuffer id_src = buffer->wgpu;
    GGML_ASSERT(id_src);

    WGPUBuffer ph = wgpuDeviceCreateBuffer(ctx->device, &(const WGPUBufferDescriptor){
                                                            .label = "ph",
                                                            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
                                                            .size = size_aligned,
                                                            .mappedAtCreation = false,
                                                         });

    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(
            ctx->device, &(const WGPUCommandEncoderDescriptor){
                        .label = "ggml_command_encoder_read_back_buffer",
                    });
    GGML_ASSERT(command_encoder);


    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, id_src, 0,
                                        ph, 0, original_size);

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(
        command_encoder, &(const WGPUCommandBufferDescriptor){
                            .label = "command_buffer_read_back_buffer",
                        });
    GGML_ASSERT(command_buffer);

    wgpuQueueSubmit(ctx->queue, 1, &command_buffer);

    wgpuBufferMapAsync(ph, WGPUMapMode_Read, 0, original_size,
                     handle_buffer_map, ctx);
    wait_for_buffer_map(ctx);
    // wgpuDevicePoll(ctx->device, true, NULL);

    const void * buf = wgpuBufferGetConstMappedRange(ph, 0, original_size);
    GGML_ASSERT(buf);

    memcpy(buffer->data, buf, original_size);
    wgpuBufferUnmap(ph);

    wgpuCommandBufferRelease(command_buffer);
    wgpuCommandEncoderRelease(command_encoder);
    wgpuBufferDestroy(ph);
    wgpuBufferRelease(ph);
}


void ggml_wgpu_write_buffer(
        struct ggml_wgpu_context * ctx,
                       const char * name) {
    struct ggml_wgpu_buffer * buffer = NULL;
    for (int i = 0; i < ctx->n_buffers; ++i) {
        if (strcmp(ctx->buffers[i].name, name) == 0) {
            GGML_ASSERT(buffer == NULL); // this function does not yet support reading split buffers
            buffer = &ctx->buffers[i];
        }
    }

    GGML_ASSERT(buffer);

    wgpuQueueWriteBuffer(ctx->queue, buffer->wgpu, 0, buffer->data, buffer->size);
}

void ggml_wgpu_set_tensor(
        struct ggml_wgpu_context * ctx,
        struct ggml_tensor * t) {
    size_t offs;
    WGPUBuffer id_dst = ggml_wgpu_get_buffer(ctx, t, &offs);
    GGML_ASSERT(id_dst);

    GGML_ASSERT(ggml_is_contiguous(t));
    wgpuQueueWriteBuffer(ctx->queue, id_dst, offs, t->data, ggml_nbytes(t));
}

void ggml_wgpu_get_tensor(
        struct ggml_wgpu_context * ctx,
        struct ggml_tensor * t) {
    GGML_ASSERT(ggml_is_contiguous(t));
    size_t offs;
    WGPUBuffer id_src = ggml_wgpu_get_buffer(ctx, t, &offs);
    GGML_ASSERT(id_src);

    const size_t nbytes = ggml_nbytes(t);

    WGPUBuffer ph = wgpuDeviceCreateBuffer(ctx->device, &(const WGPUBufferDescriptor){
                                                            .label = "ph",
                                                            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
                                                            .size = nbytes,
                                                            .mappedAtCreation = false,
                                                         });

    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(
            ctx->device, &(const WGPUCommandEncoderDescriptor){
                        .label = "ggml_command_encoder_get_tensor",
                    });
    GGML_ASSERT(command_encoder);


    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, id_src, offs,
                                        ph, 0, nbytes);

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(
        command_encoder, &(const WGPUCommandBufferDescriptor){
                            .label = "command_buffer_get_tensor",
                        });
    GGML_ASSERT(command_buffer);

    wgpuQueueSubmit(ctx->queue, 1, &command_buffer);

    wgpuBufferMapAsync(ph, WGPUMapMode_Read, 0, nbytes,
                     handle_buffer_map, ctx);
    wait_for_buffer_map(ctx);
    // wgpuDevicePoll(ctx->device, true, NULL);

    const void * buf = wgpuBufferGetConstMappedRange(ph, 0, nbytes);
    GGML_ASSERT(buf);

    memcpy(t->data, buf, nbytes);
    wgpuBufferUnmap(ph);

    wgpuCommandBufferRelease(command_buffer);
    wgpuCommandEncoderRelease(command_encoder);
    wgpuBufferDestroy(ph);
    wgpuBufferRelease(ph);
}

void ggml_wgpu_graph_compute(
        struct ggml_wgpu_context * ctx,
               struct ggml_cgraph * gf) {

    const int64_t perf_start_cycles  = ggml_perf_cycles();
    const int64_t perf_start_time_us = ggml_perf_time_us();

    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(
            ctx->device, &(const WGPUCommandEncoderDescriptor){
                        .label = "ggml_command_encoder",
                    });
    GGML_ASSERT(command_encoder);


    for (int i = 0; i < gf->n_nodes; ++i) {
        // GGML_WGPU_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));
        if (ctx->timestamp_queries) {
            wgpuCommandEncoderWriteTimestamp(command_encoder, ctx->timestamp_queries, i);
        }

        struct ggml_tensor * dst  = gf->nodes[i];
        GGML_ASSERT(dst);

        switch (dst->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
                {
                    continue;
                } break;
            default:
                {
                }
        }

        // GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#if MY_OPTI_USE_F16
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
#else
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
        size_t offs_dst  = 0;
        WGPUBuffer id_dst  = ggml_wgpu_get_buffer(ctx, dst,  &offs_dst);
        GGML_ASSERT(id_dst);
        ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].ne[0] = dst->ne[0];
        ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].ne[1] = dst->ne[1];
        ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].ne[2] = dst->ne[2];
        ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].ne[3] = dst->ne[3];
        ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].nb[0] = dst->nb[0]/ggml_element_size(dst);
        ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].nb[1] = dst->nb[1]/ggml_element_size(dst);
        ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].nb[2] = dst->nb[2]/ggml_element_size(dst);
        ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].nb[3] = dst->nb[3]/ggml_element_size(dst);
        ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].offset = 0*offs_dst/ggml_element_size(dst);

        ctx->bind_group_entries[GGML_WGPU_DST_BINDING_INDEX].binding = GGML_WGPU_DST_BINDING_INDEX;
        ctx->bind_group_entries[GGML_WGPU_DST_BINDING_INDEX].buffer = id_dst;
        ctx->bind_group_entries[GGML_WGPU_DST_BINDING_INDEX].offset = offs_dst;
        ctx->bind_group_entries[GGML_WGPU_DST_BINDING_INDEX].size = ggml_nbytes(dst);

        GGML_ASSERT(ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[GGML_WGPU_DST_BINDING_INDEX].nb[0] == 1);
        // GGML_ASSERT(0 == (dst->nb[1]%16));

        int is_op_inplace_which_src = -1;

        for (int src_idx=0; src_idx < GGML_MAX_SRC; ++src_idx) {
            struct ggml_tensor * srci = dst->src[src_idx];
            if (srci && dst->data == srci->data) {
                // CASE of aliasing of src0 and dst (in-place operation) - we need to use the same buffer for both, as WGSL does not allow aliasing of writeable buffers
                GGML_ASSERT(src_idx == 0 && is_op_inplace_which_src == -1 && (dst->op == GGML_OP_SCALE || dst->op == GGML_OP_SPECIAL_ADAM_STEP));
                is_op_inplace_which_src = src_idx;
                srci = NULL;
            }
            // if (srci) GGML_ASSERT(0 == (srci->nb[1]%16));

            const enum ggml_type srcit = srci ? srci->type : GGML_TYPE_COUNT;
            // GGML_ASSERT(srcit == GGML_TYPE_F32 || srcit == GGML_TYPE_F16 || srcit == GGML_TYPE_COUNT);
#if MY_OPTI_USE_F16
            if (dst->op == GGML_OP_SPECIAL_ADAM_STEP && (src_idx == 2 || src_idx == 3 || src_idx == 4)) {
                GGML_ASSERT(srcit == GGML_TYPE_F32);
            } else {
                GGML_ASSERT(srcit == GGML_TYPE_F16 || srcit == GGML_TYPE_COUNT);
            }
#else
            GGML_ASSERT(srcit == GGML_TYPE_F32 || srcit == GGML_TYPE_COUNT);
#endif

            size_t offs_srci = 0;
            WGPUBuffer id_srci = srci ? ggml_wgpu_get_buffer(ctx, srci, &offs_srci) : NULL;

            ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].ne[0] = srci ? srci->ne[0] : 1;
            ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].ne[1] = srci ? srci->ne[1] : 1;
            ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].ne[2] = srci ? srci->ne[2] : 1;
            ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].ne[3] = srci ? srci->ne[3] : 1;
            ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].nb[0] = srci ? srci->nb[0]/ggml_element_size(srci) : 1;
            ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].nb[1] = srci ? srci->nb[1]/ggml_element_size(srci) : 1;
            ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].nb[2] = srci ? srci->nb[2]/ggml_element_size(srci) : 1;
            ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].nb[3] = srci ? srci->nb[3]/ggml_element_size(srci) : 1;
            ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].offset = srci ? 0*offs_srci/ggml_element_size(srci) : 0;

            ctx->bind_group_entries[src_idx].binding = src_idx;
            ctx->bind_group_entries[src_idx].buffer = id_srci ? id_srci : ctx->placeholder_buffer[src_idx];
            ctx->bind_group_entries[src_idx].offset = id_srci ? offs_srci : 0;
            ctx->bind_group_entries[src_idx].size = id_srci ? ggml_nbytes(srci) : PLACEHOLDER_BUFFER_SIZE;

            GGML_ASSERT(ctx->tensor_dimension_operation_params_host[i].tensor_dimension_params[src_idx].nb[0] == 1);
        }

        for (int op_par_idx=0; op_par_idx < (GGML_MAX_OP_PARAMS/sizeof(int32_t)); ++op_par_idx) {
            ctx->tensor_dimension_operation_params_host[i].op_params[op_par_idx] = dst->op_params[op_par_idx];
        }

        for (int extra_uniform_idx=0; extra_uniform_idx < GGML_WGPU_NUM_EXTRA_UNIFORM_BINDINGS; ++extra_uniform_idx) {
            ctx->bind_group_entries[GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + extra_uniform_idx].binding = GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + extra_uniform_idx;
            ctx->bind_group_entries[GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + extra_uniform_idx].buffer = ctx->placeholder_uniform_buffer[extra_uniform_idx];
            ctx->bind_group_entries[GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + extra_uniform_idx].offset = i*GGML_WGPU_EXTRA_UNIFORM_SIZE;
            ctx->bind_group_entries[GGML_WGPU_FIRST_EXTRA_UNIFORM_INDEX + extra_uniform_idx].size = GGML_WGPU_EXTRA_UNIFORM_SIZE;
        }

        ctx->bind_group_entries[GGML_WGPU_DIM_PARAMS_BINDING_INDEX].offset = i * GGML_WGPU_OPERATOR_PARAMS_SIZE_TOTAL;
        wgpuQueueWriteBuffer(ctx->queue, ctx->tensor_dimension_operation_params, i * GGML_WGPU_OPERATOR_PARAMS_SIZE_TOTAL, 
            &(ctx->tensor_dimension_operation_params_host[i]), sizeof(ctx->tensor_dimension_operation_params_host[i]));

        char bind_group_name[30];
        char compute_pass_name[30];
        sprintf(bind_group_name, "bind_group_%d", i);
        sprintf(compute_pass_name, "compute_pass_%d", i);

        WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
            ctx->device, &(const WGPUBindGroupDescriptor){
                        .label = bind_group_name,
                        .layout = ctx->bind_group_layout,
                        .entryCount = GGML_WGPU_BINDINGS_SIZE,
                        .entries = ctx->bind_group_entries,
                    });
        GGML_ASSERT(bind_group);

        WGPUComputePassEncoder compute_pass_encoder = NULL;


        //GGML_METAL_LOG_INFO("%s: op - %s\n", __func__, ggml_op_name(dst->op));
        //if (src0) {
        //    GGML_METAL_LOG_INFO("%s: src0 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src0t), ne00, ne01, ne02,
        //            ggml_is_contiguous(src0), src0->name);
        //}
        //if (src1) {
        //    GGML_METAL_LOG_INFO("%s: src1 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src1t), ne10, ne11, ne12,
        //            ggml_is_contiguous(src1), src1->name);
        //}
        //if (dst) {
        //    GGML_METAL_LOG_INFO("%s: dst  - %4s [%5lld, %5lld, %5lld], 1, %s\n",  __func__, ggml_type_name(dstt),  ne0,  ne1,  ne2,
        //            dst->name);
        //}

        #define GGML_WGPU_ENCODE_KERNEL(name, wcX, wcY, wcZ) \
            compute_pass_encoder = wgpuCommandEncoderBeginComputePass( \
                command_encoder, &(const WGPUComputePassDescriptor){    \
                                    .label = compute_pass_name,          \
                                });                                   \
            GGML_ASSERT(compute_pass_encoder);                        \
            wgpuComputePassEncoderSetPipeline(compute_pass_encoder, ctx->pipeline_##name); \
            wgpuComputePassEncoderSetBindGroup(compute_pass_encoder, 0, bind_group, 0, NULL); \
            wgpuComputePassEncoderDispatchWorkgroups(compute_pass_encoder, (wcX), (wcY), (wcZ)); \
            wgpuComputePassEncoderEnd(compute_pass_encoder);

        switch (dst->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
                {
                    // noop
                } break;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(gf->nodes[i])) {
                    case GGML_UNARY_OP_SILU:
                        {
                            GGML_WGPU_ENCODE_KERNEL(silu, ggml_nelements(dst), 1, 1)
                        } break;
                    default:
                        {
                            GGML_WGPU_LOG_WARN("%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                            GGML_ASSERT(false);
                        }
                } break;
            case GGML_OP_CONV_1D_SMALL_KERN:
                {
                    GGML_ASSERT(1 == dst->op_params[0]); // stride
                    GGML_ASSERT(0 == dst->op_params[1]); // padding

                    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, ctx->bind_group_entries[0].buffer, ctx->bind_group_entries[0].offset,
                                                         ctx->placeholder_uniform_buffer[0], i*GGML_WGPU_EXTRA_UNIFORM_SIZE, ctx->bind_group_entries[0].size);
                    if (dst->src[2]) {
                        wgpuCommandEncoderCopyBufferToBuffer(command_encoder, ctx->bind_group_entries[2].buffer, ctx->bind_group_entries[2].offset,
                                                            ctx->placeholder_uniform_buffer[1], i*GGML_WGPU_EXTRA_UNIFORM_SIZE, ctx->bind_group_entries[2].size);
                    }
                    const int32_t d0 = dst->op_params[2];
                    const int32_t num_threads_x = 16;
                    const int32_t vals_per_thread = 16;
                    const int64_t nk = dst->src[0]->ne[2];
                    const int64_t output_len = dst->ne[0];
                    const int64_t real_input_len = output_len + d0*(nk-1);
                    if (1 == nk) {
                        const int32_t dispatch_x = CEIL_DIV(output_len, 256);
                        GGML_ASSERT(0 == dst->op_params[3]);
                        GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_simpl, dispatch_x, dst->ne[1], dst->ne[2])
                    } else {
                        GGML_ASSERT(real_input_len == dst->src[1]->ne[0]);
                        if (dst->src[3]) {
                            GGML_ASSERT(output_len == dst->src[3]->ne[0]);
                        }
                        if (0 && d0 >=16 && 16 == dst->src[0]->ne[0] && 16 == dst->src[0]->ne[1] && 3 == nk) {
                            GGML_ASSERT(false);
                            // const int32_t vals_to_round = MAX(d0, num_threads_x);
                            // const int32_t dispatch_x = vals_to_round/num_threads_x * CEIL_DIV(output_len, (vals_to_round * vals_per_thread));
                            // const int32_t dispatch_y = 1;//CEIL_DIV(dst->ne[1], 16);
                            // if (d0 >=16) {
                            //     GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_opti_large_dil, dispatch_x, dispatch_y, dst->ne[2])
                            // } else {
                            //     GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_opti, dispatch_x, dispatch_y, dst->ne[2])
                            // }
                        } else {
                            if (d0>=4) {
                                if (0 && dst->ne[1] == 8) {
                                    const int32_t dispatch_x = CEIL_DIV(output_len, 4*32*16);
                                    // GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_no_offsets_8x8x3, dispatch_x, 1, dst->ne[2])
                                } else {
                                    const int32_t dispatch_x = CEIL_DIV(output_len, 4*256);
                                    const int32_t dispatch_y = dst->ne[1];
                                    GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_no_offsets, dispatch_x, dispatch_y, dst->ne[2])
                                }
                            } else {
                                const int32_t dispatch_x = CEIL_DIV(output_len, 256);
                                const int32_t dispatch_y = dst->ne[1];
                                GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern, dispatch_x, dispatch_y, dst->ne[2])
                            }
                            // const int32_t dispatch_x = CEIL_DIV(output_len, 4*256);
                            // // const int32_t dispatch_y = CEIL_DIV(dst->ne[1], 2);
                            // const int32_t dispatch_y = dst->ne[1];
                            // // GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern, dispatch_x, dispatch_y, dst->ne[2])
                            // GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_no_offsets, dispatch_x, dispatch_y, dst->ne[2])
                        }
                    }
                } break;
            case GGML_OP_ADD_AND_TRIM:
                {
                    const int32_t dispatch_x = CEIL_DIV(dst->ne[0], 256);
                    GGML_ASSERT(dst->ne[3] == 1);
                    GGML_WGPU_ENCODE_KERNEL(add_and_trim, dispatch_x, dst->ne[1], dst->ne[2])
                } break;
            case GGML_OP_SCALE:
                {
                    GGML_ASSERT(ggml_is_contiguous(dst));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
                    const int32_t dispatch_x = CEIL_DIV(ggml_nelements_padded(dst), 256);
                    if (is_op_inplace_which_src == 0) {
                        GGML_WGPU_ENCODE_KERNEL(scale_inplace, dispatch_x, 1, 1)
                    } else {
                        GGML_WGPU_ENCODE_KERNEL(scale, dispatch_x, 1, 1)
                    }
                } break;
            case GGML_OP_SUB:
                {
                    GGML_ASSERT(ggml_is_contiguous(dst));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
                    const int32_t dispatch_x = CEIL_DIV(ggml_nelements_padded(dst), 256);
                    GGML_WGPU_ENCODE_KERNEL(sub, dispatch_x, 1, 1)
                } break;
            case GGML_OP_SQR:
                {
                    GGML_ASSERT(ggml_is_contiguous(dst));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
                    const int32_t dispatch_x = CEIL_DIV(ggml_nelements_padded(dst), 256);
                    GGML_WGPU_ENCODE_KERNEL(sqr, dispatch_x, 1, 1)
                } break;
            case GGML_OP_SUM:
                {
                    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
                    GGML_WGPU_ENCODE_KERNEL(sum, 1, 1, 1)
                } break;
            case GGML_OP_REPEAT:
                {
                    const int32_t dispatch_x = CEIL_DIV(dst->ne[0], 256);
                    GGML_ASSERT(dst->ne[3] == 1);
                    GGML_WGPU_ENCODE_KERNEL(repeat, dispatch_x, dst->ne[1], dst->ne[2])
                } break;
            case GGML_OP_MUL:
                {
                    const int32_t dispatch_x = CEIL_DIV(dst->ne[0], 256);
                    GGML_ASSERT(dst->ne[3] == 1);
                    GGML_WGPU_ENCODE_KERNEL(mul, dispatch_x, dst->ne[1], dst->ne[2])
                } break;
            case GGML_OP_CONV_1D_SMALL_KERN_BACK_FILTER:
                {
                    GGML_ASSERT(dst->ne[3] == 1);
#if 0
                    GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_filter, dst->ne[2], dst->ne[0], dst->ne[1])
#else
                    if ((dst->ne[2] > 1) || (dst->ne[0]*dst->ne[1] > 32)) {
                        if (dst->ne[2] > 1) {
                            GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_filter, dst->ne[2], dst->ne[0], dst->ne[1])
                        } else {
                            GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_filter_nk1, dst->ne[0], dst->ne[1], 1)
                        }
                    } else {
                        GGML_ASSERT((ggml_nbytes(dst)*dst->src[0]->ne[2]) <= PLACEHOLDER_BUFFER_SIZE);
                        GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_filter_stage1, dst->ne[2], dst->ne[0], dst->ne[1]*dst->src[0]->ne[2])
                        GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_filter_stage2, dst->ne[2], dst->ne[0], dst->ne[1])
                    }
#endif
                } break;
            case GGML_OP_CONV_1D_SMALL_KERN_BACK_INPUT:
                {
                    const int32_t d0 = dst->op_params[2];
                    GGML_ASSERT(dst->ne[3] == 1);
                    if (d0 >= 4) {
                        const int32_t dispatch_x = CEIL_DIV(dst->ne[0], 4*256);
                        GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_input_large_dil, dispatch_x, dst->ne[1], dst->ne[2])
                    } else {
                        const int32_t dispatch_x = CEIL_DIV(dst->ne[0], 256);
                        GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_input, dispatch_x, dst->ne[1], dst->ne[2])
                    }
                } break;
            case GGML_OP_ACC:
                {
                    const int32_t dispatch_x = CEIL_DIV(dst->ne[0], 256);
                    GGML_ASSERT(dst->ne[3] == 1);
                    GGML_WGPU_ENCODE_KERNEL(acc, dispatch_x, dst->ne[1], dst->ne[2])
                } break;
            case GGML_OP_ADD_AND_TANH_BACK:
                {
                    GGML_ASSERT(ggml_is_contiguous(dst));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

                    const int32_t dispatch_x = CEIL_DIV(ggml_nelements_padded(dst), 4*256);
                    GGML_ASSERT(dst->ne[3] == 1);
                    GGML_WGPU_ENCODE_KERNEL(add_and_tanh_back, dispatch_x, 1, 1)
                } break;
            case GGML_OP_ADD:
                {
                    const int32_t dispatch_x = CEIL_DIV(dst->ne[0], 256);
                    GGML_ASSERT(dst->ne[3] == 1);
                    GGML_WGPU_ENCODE_KERNEL(add, dispatch_x, dst->ne[1], dst->ne[2])
                } break;
            case GGML_OP_CONV_1D_SMALL_KERN_BACK_BIAS:
                {
                    GGML_ASSERT(dst->ne[0] == 1);
                    GGML_ASSERT(dst->ne[2] == 1);
                    GGML_ASSERT(dst->ne[3] == 1);
#if 0
                    GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_bias, dst->ne[1], 1, 1)
#else
                    GGML_ASSERT((4*dst->ne[1]*dst->src[0]->ne[2] *ggml_element_size(dst)) <= PLACEHOLDER_BUFFER_SIZE);
                    GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_bias_stage1, dst->ne[1], dst->src[0]->ne[2], 1)
                    GGML_WGPU_ENCODE_KERNEL(conv_1d_small_kern_back_bias_stage2, dst->ne[1], 1, 1)
#endif
                } break;
            case GGML_OP_SPECIAL_ADAM_STEP:
                {
                    GGML_ASSERT(ggml_is_contiguous(dst));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[2]));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[3]));
                    GGML_ASSERT(ggml_is_contiguous(dst->src[4]));
                    const int32_t dispatch_x = CEIL_DIV(ggml_nelements_padded(dst), 256);
                    if (is_op_inplace_which_src == 0) {
                        GGML_WGPU_ENCODE_KERNEL(special_adam_step_inplace, dispatch_x, 1, 1)
                    } else {
                        GGML_WGPU_ENCODE_KERNEL(special_adam_step, dispatch_x, 1, 1)
                    }
                } break;
            default:
                {
                    GGML_WGPU_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                    GGML_ASSERT(false);
                }
        }

        #undef GGML_WGPU_ENCODE_KERNEL

        if (bind_group) wgpuBindGroupRelease(bind_group);
        if (compute_pass_encoder) wgpuComputePassEncoderRelease(compute_pass_encoder);
    }

    if (ctx->timestamp_queries) {
        wgpuCommandEncoderWriteTimestamp(command_encoder, ctx->timestamp_queries, gf->n_nodes);
        wgpuCommandEncoderResolveQuerySet(command_encoder, ctx->timestamp_queries, 0, gf->n_nodes + 1, ctx->timestamp_queries_resolve_buffer, 0);
        wgpuCommandEncoderCopyBufferToBuffer(command_encoder, ctx->timestamp_queries_resolve_buffer, 0, 
            ctx->timestamp_queries_read_buffer, 0, 8*(gf->n_nodes + 1));
    }

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(
                                        command_encoder, &(const WGPUCommandBufferDescriptor){
                                                            .label = "command_buffer",
                       });
    GGML_ASSERT(command_buffer);

    wgpuQueueSubmit(ctx->queue, 1, &command_buffer);

    wgpuCommandBufferRelease(command_buffer);
    wgpuCommandEncoderRelease(command_encoder);

    if (ctx->timestamp_queries) {
        wgpuBufferMapAsync(ctx->timestamp_queries_read_buffer, WGPUMapMode_Read, 0, 8*(gf->n_nodes + 1),
                        handle_buffer_map, ctx);
        wait_for_buffer_map(ctx);
        // wgpuDevicePoll(ctx->device, true, NULL);

        const void * buf = wgpuBufferGetConstMappedRange(ctx->timestamp_queries_read_buffer, 0, 8*(gf->n_nodes + 1));
        GGML_ASSERT(buf);

        uint64_t * timestamps = (uint64_t *) buf;
        for (int i = 0; i < gf->n_nodes; ++i) {
            gf->nodes[i]->perf_runs++;
            gf->nodes[i]->perf_cycles += timestamps[i + 1] - timestamps[i];
            gf->nodes[i]->perf_time_us += (timestamps[i + 1] - timestamps[i]) / 1000;
        }
        

        wgpuBufferUnmap(ctx->timestamp_queries_read_buffer);
    }

#ifdef WEBGPU_BACKEND_WGPU
    wgpuDevicePoll(ctx->device, true, NULL);
#endif
    // performance stats (graph)
    {
        int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_start_cycles;
        int64_t perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

        gf->perf_runs++;
        gf->perf_cycles  += perf_cycles_cur;
        gf->perf_time_us += perf_time_us_cur;

        GGML_WGPU_LOG_INFO("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n",
                __func__, gf->perf_runs,
                (double) perf_cycles_cur      / (double) ggml_cycles_per_ms(),
                (double) gf->perf_cycles  / (double) ggml_cycles_per_ms() / (double) gf->perf_runs,
                (double) perf_time_us_cur     / 1000.0,
                (double) gf->perf_time_us / 1000.0 / gf->perf_runs);
    }

}


////////////////////////////////////////////////////////////////////////////////

// backend interface

static const char * ggml_backend_wgpu_name(ggml_backend_t backend) {
    return "WebGPU";

    UNUSED(backend);
}

static void ggml_backend_wgpu_free(ggml_backend_t backend) {
    struct ggml_wgpu_context * ctx = (struct ggml_wgpu_context *)backend->context;
    ggml_wgpu_free(ctx);
    free(backend);
}

static void * ggml_backend_wgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)buffer->context;
}

static void ggml_backend_wgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
    UNUSED(buffer);
}

static struct ggml_backend_buffer_i wgpu_backend_buffer_i = {
    /* .free_buffer    = */ ggml_backend_wgpu_buffer_free_buffer,
    /* .get_base       = */ ggml_backend_wgpu_buffer_get_base,
    /* .get_alloc_size = */ NULL, // defaults to ggml_nbytes
    /* .init_tensor    = */ NULL, // no initialization required
    /* .free_tensor    = */ NULL, // no cleanup required
};

static ggml_backend_buffer_t ggml_backend_wgpu_alloc_buffer(ggml_backend_t backend, size_t size) {
    struct ggml_wgpu_context * ctx = (struct ggml_wgpu_context *)backend->context;

    void * data = ggml_wgpu_host_malloc(size);

    // TODO: set proper name of the buffers
    ggml_wgpu_add_buffer(ctx, "backend", data, size, 0);

    return ggml_backend_buffer_init(backend, wgpu_backend_buffer_i, data, size);
}

static size_t ggml_backend_wgpu_get_alignment(ggml_backend_t backend) {
    return 32;
    UNUSED(backend);
}

static void ggml_backend_wgpu_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(ggml_get_backend(tensor) == backend && "tensor not allocated on this backend");

    memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_wgpu_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(ggml_get_backend(tensor) == backend && "tensor not allocated on this backend");

    memcpy(data, (const char *)tensor->data + offset, size);
}

static void ggml_backend_wgpu_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
}

static void ggml_backend_wgpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_wgpu_context * wgpu_ctx = (struct ggml_wgpu_context *)backend->context;

    ggml_wgpu_graph_compute(wgpu_ctx, cgraph);
}

static bool ggml_backend_wgpu_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return true;
    UNUSED(backend);
    UNUSED(op);
}

static struct ggml_backend_i wgpu_backend_i = {
    /* .get_name            = */ ggml_backend_wgpu_name,
    /* .free                = */ ggml_backend_wgpu_free,
    /* .alloc_buffer        = */ ggml_backend_wgpu_alloc_buffer,
    /* .get_alignment       = */ ggml_backend_wgpu_get_alignment,
    /* .set_tensor_async    = */ ggml_backend_wgpu_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_wgpu_get_tensor_async,
    /* .synchronize         = */ ggml_backend_wgpu_synchronize,
    /* .cpy_tensor_from     = */ NULL,
    /* .cpy_tensor_to       = */ NULL,
    /* .graph_plan_create   = */ NULL, // the wgpu implementation does not require creating graph plans atm
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_wgpu_graph_compute,
    /* .supports_op         = */ ggml_backend_wgpu_supports_op,
};

ggml_backend_t ggml_backend_wgpu_init(void) {
    struct ggml_wgpu_context * ctx = malloc(sizeof(struct ggml_wgpu_context));

    ctx = ggml_wgpu_init();

    ggml_backend_t wgpu_backend = malloc(sizeof(struct ggml_backend));

    *wgpu_backend = (struct ggml_backend) {
        /* .interface = */ wgpu_backend_i,
        /* .context   = */ ctx,
    };

    return wgpu_backend;
}

bool ggml_backend_is_wgpu(ggml_backend_t backend) {
    return backend->iface.get_name == ggml_backend_wgpu_name;
}

