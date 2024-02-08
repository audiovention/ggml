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
var<storage,read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage,read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage,read_write> src2: array<f32>;

@group(0) @binding(3)
var<storage,read_write> src3: array<f32>;

@group(0) @binding(4)
var<storage,read_write> src4: array<f32>;

@group(0) @binding(5)
var<storage,read_write> src5: array<f32>;

@group(0) @binding(6)
var<storage,read_write> dst: array<f32>;


@group(0) @binding(0)
var<storage,read_write> src0_v4: array<vec4f>;

@group(0) @binding(1)
var<storage,read_write> src1_v4: array<vec4f>;

@group(0) @binding(2)
var<storage,read_write> src2_v4: array<vec4f>;

@group(0) @binding(3)
var<storage,read_write> src3_v4: array<vec4f>;

@group(0) @binding(4)
var<storage,read_write> src4_v4: array<vec4f>;

@group(0) @binding(5)
var<storage,read_write> src5_v4: array<vec4f>;

@group(0) @binding(6)
var<storage,read_write> dst_v4: array<vec4f>;



@group(0) @binding(7)
var<uniform> tensor_dimension_params: TensorDimensionParams;


@group(0) @binding(8)
var<uniform> extra_uniform0: array<vec4f, 1024>;

@group(0) @binding(9)
var<uniform> extra_uniform1: array<vec4f, 1024>;


var<workgroup> workgroup_data: array<f32, 256>;
var<workgroup> workgroup_data_v4f: array<vec4f, 256>;


fn get_src0(x: u32, y: u32, z: u32) -> f32 {
    return src0[x 
                //   * tensor_dimension_params.src[0].nb[0]
                + y * tensor_dimension_params.src[0].nb[1] +
                z * tensor_dimension_params.src[0].nb[2]
                    // + tensor_dimension_params.src[0].offset
                    ];
}

fn get_src1(x: u32, y: u32, z: u32) -> f32 {
    return src1[x 
                //   * tensor_dimension_params.src[1].nb[0]
                + y * tensor_dimension_params.src[1].nb[1] +
                z * tensor_dimension_params.src[1].nb[2]
                    // + tensor_dimension_params.src[1].offset
                    ];
}

fn get_src2(x: u32, y: u32, z: u32) -> f32 {
    return src2[x 
                //   * tensor_dimension_params.src[2].nb[0]
                + y * tensor_dimension_params.src[2].nb[1] +
                z * tensor_dimension_params.src[2].nb[2]
                    // + tensor_dimension_params.src[2].offset
                    ];
}

fn get_src3(x: u32, y: u32, z: u32) -> f32 {
    return src3[x 
                //   * tensor_dimension_params.src[3].nb[0]
                + y * tensor_dimension_params.src[3].nb[1] +
                z * tensor_dimension_params.src[3].nb[2]
                    // + tensor_dimension_params.src[3].offset
                    ];
}

fn get_src4(x: u32, y: u32, z: u32) -> f32 {
    return src4[x 
                //   * tensor_dimension_params.src[4].nb[0]
                + y * tensor_dimension_params.src[4].nb[1] +
                z * tensor_dimension_params.src[4].nb[2]
                    // + tensor_dimension_params.src[4].offset
                    ];
}

fn get_src5(x: u32, y: u32, z: u32) -> f32 {
    return src5[x 
                //   * tensor_dimension_params.src[5].nb[0]
                + y * tensor_dimension_params.src[5].nb[1] +
                z * tensor_dimension_params.src[5].nb[2]
                    // + tensor_dimension_params.src[5].offset
                    ];
}

fn set_dst(x: u32, y: u32, z: u32, v: f32) {
    dst[ x 
        //    * tensor_dimension_params.dst.nb[0]
         + y * tensor_dimension_params.dst.nb[1] +
         z * tensor_dimension_params.dst.nb[2]
            // + tensor_dimension_params.dst.offset
             ] = v;
}


fn get_src0_lin(x: u32) -> f32 {
    return src0[x 
                //   * tensor_dimension_params.src[0].nb[0]
                    // + tensor_dimension_params.src[0].offset
                    ];
}

fn get_src1_lin(x: u32) -> f32 {
    return src1[x 
                //   * tensor_dimension_params.src[1].nb[0]
                    // + tensor_dimension_params.src[1].offset
                    ];
}

fn get_src2_lin(x: u32) -> f32 {
    return src2[x 
                //   * tensor_dimension_params.src[2].nb[0]
                    // + tensor_dimension_params.src[2].offset
                    ];
}

fn get_src3_lin(x: u32) -> f32 {
    return src3[x 
                //   * tensor_dimension_params.src[3].nb[0]
                    // + tensor_dimension_params.src[3].offset
                    ];
}

fn get_src4_lin(x: u32) -> f32 {
    return src4[x 
                //   * tensor_dimension_params.src[4].nb[0]
                    // + tensor_dimension_params.src[4].offset
                    ];
}

fn get_src5_lin(x: u32) -> f32 {
    return src5[x 
                //   * tensor_dimension_params.src[5].nb[0]
                    // + tensor_dimension_params.src[5].offset
                    ];
}

fn get_dst_lin(x: u32) -> f32 {
    return dst[ x 
        //    * tensor_dimension_params.dst.nb[0]
            // + tensor_dimension_params.dst.offset
             ];
}


fn set_dst_lin(x: u32, v: f32) {
    dst[ x 
        //    * tensor_dimension_params.dst.nb[0]
            // + tensor_dimension_params.dst.offset
             ] = v;
}

fn set_src0_lin(x: u32, v: f32) {
    src0[ x 
        //    * tensor_dimension_params.src[0].nb[0]
            // + tensor_dimension_params.src[0].offset
             ] = v;
}

fn set_src1_lin(x: u32, v: f32) {
    src1[ x 
        //    * tensor_dimension_params.src[1].nb[0]
            // + tensor_dimension_params.src[1].offset
             ] = v;
}

fn set_src2_lin(x: u32, v: f32) {
    src2[ x 
        //    * tensor_dimension_params.src[2].nb[0]
            // + tensor_dimension_params.src[2].offset
             ] = v;
}

fn set_src3_lin(x: u32, v: f32) {
    src3[ x 
        //    * tensor_dimension_params.src[3].nb[0]
            // + tensor_dimension_params.src[3].offset
             ] = v;
}

fn set_src4_lin(x: u32, v: f32) {
    src4[ x 
        //    * tensor_dimension_params.src[4].nb[0]
            // + tensor_dimension_params.src[4].offset
             ] = v;
}

fn set_src5_lin(x: u32, v: f32) {
    src5[ x 
        //    * tensor_dimension_params.src[5].nb[0]
            // + tensor_dimension_params.src[5].offset
             ] = v;
}


fn set_src0(x: u32, y: u32, z: u32, v: f32) {
    src0[ x 
        //    * tensor_dimension_params.src[0].nb[0]
         + y * tensor_dimension_params.src[0].nb[1] +
         z * tensor_dimension_params.src[0].nb[2]
            // + tensor_dimension_params.src[0].offset
             ] = v;
}

fn set_src1(x: u32, y: u32, z: u32, v: f32) {
    src1[ x 
        //    * tensor_dimension_params.src[1].nb[0]
         + y * tensor_dimension_params.src[1].nb[1] +
         z * tensor_dimension_params.src[1].nb[2]
            // + tensor_dimension_params.src[1].offset
             ] = v;
}

fn set_src2(x: u32, y: u32, z: u32, v: f32) {
    src2[ x 
        //    * tensor_dimension_params.src[2].nb[0]
         + y * tensor_dimension_params.src[2].nb[1] +
         z * tensor_dimension_params.src[2].nb[2]
            // + tensor_dimension_params.src[2].offset
             ] = v;
}

fn set_src3(x: u32, y: u32, z: u32, v: f32) {
    src3[ x 
        //    * tensor_dimension_params.src[3].nb[0]
         + y * tensor_dimension_params.src[3].nb[1] +
         z * tensor_dimension_params.src[3].nb[2]
            // + tensor_dimension_params.src[3].offset
             ] = v;
}

fn set_src4(x: u32, y: u32, z: u32, v: f32) {
    src4[ x 
        //    * tensor_dimension_params.src[4].nb[0]
         + y * tensor_dimension_params.src[4].nb[1] +
         z * tensor_dimension_params.src[4].nb[2]
            // + tensor_dimension_params.src[4].offset
             ] = v;
}

fn set_src5(x: u32, y: u32, z: u32, v: f32) {
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

@compute
@workgroup_size(1)
fn kernel_silu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x =  get_src0_lin(global_id.x);
    set_dst_lin(global_id.x, x / (1.0 + exp(-x)));
}


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
        let bias_idx = global_id.y * tensor_dimension_params.src[2].nb[1];
        let bias = extra_uniform1[bias_idx/4u][bias_idx%4u];
        // let bias = get_src2(0u, global_id.y, 0u);
        output += bias;
    }

    if (has_inject_signal) {
        output += get_src3(u32(tensor_dimension_params.src[3].ne[0]) - output_len + global_id.x, global_id.y, global_id.z);
    }

    let base_src1_offset = input_len - real_input_len + global_id.x + global_id.z * tensor_dimension_params.src[1].nb[2];

    for (var ik = 0u; ik < nk; ik = ik + 1u) {
        let in_idx_offset = ik * d0 + base_src1_offset;
        let kernel_base_idx = global_id.y + ik * tensor_dimension_params.src[0].nb[2];
        for (var ic = 0u; ic < input_channels; ic = ic + 1u) {
            let input = get_src1_lin(in_idx_offset + ic * tensor_dimension_params.src[1].nb[1]);
            // let kernel = get_src0(global_id.y, ic, ik);
            let kernel_idx = kernel_base_idx + ic * tensor_dimension_params.src[0].nb[1];
            let kernel = extra_uniform0[kernel_idx/4u][kernel_idx%4u];
            output = output + input * kernel;
        }
    }

    if (apply_tanh) {
        output = tanh(output);
    }

    set_dst(global_id.x, global_id.y, global_id.z, output);
}


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

    // var output : f32 = 0.0;
    var output = vec4f();

    if (has_bias) {
        // let bias_idx = global_id.y * tensor_dimension_params.src[2].nb[1];
        // let bias = extra_uniform1[bias_idx/4u][bias_idx%4u];
        let bias = get_src2(0u, global_id.y, 0u);
        output += bias;
    }

    if (has_inject_signal) {
        let src3_idx = global_id.x + global_id.y * tensor_dimension_params.src[3].nb[1]/4u + global_id.z * tensor_dimension_params.src[3].nb[2]/4u;
        output += src3_v4[src3_idx];
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
            output = output + input * kernel;
        }
    }

    if (apply_tanh) {
        output = tanh(output);
    }

    let dst_idx = global_id.x + global_id.y * tensor_dimension_params.dst.nb[1]/4u + global_id.z * tensor_dimension_params.dst.nb[2]/4u;
    dst_v4[dst_idx] = output;

    // set_dst(mult_idx, global_id.y, global_id.z,    output.x);
    // set_dst(mult_idx+1u, global_id.y, global_id.z, output.y);
    // set_dst(mult_idx+2u, global_id.y, global_id.z, output.z);
    // set_dst(mult_idx+3u, global_id.y, global_id.z, output.w);
}

// const nk_8x8x3 = 3u;
// const channels_8x8x3 = 8u;
// var<workgroup> workgroup_data_input_8x8x3: array<array<vec4f, channels_8x8x3>, 32>;
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

//     var kernel = array<array<f32, channels_8x8x3>, nk_8x8x3>();
//     for (var ik = 0u; ik < nk_8x8x3; ik = ik + 1u) {
//         for (var ic = 0u; ic < channels_8x8x3; ic = ic + 1u) {
//             kernel[ik][ic] = get_src0(local_id.y, ic, ik);
//         }
//     }

//     let values_vec_this_thread = min((output_len_d4 - start_idx_d4) / d0d4 + 1u, kern_output_vec_values_per_thread);
//     let input_values_vec_this_thread = values_vec_this_thread + nk_8x8x3 - 1u;


//     var output = array<vec4f, nk_8x8x3>();

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

//         var src3_here = vec4f();
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

//         output[reg_idx] = vec4f();
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

// var<workgroup> workgroup_data_kernel: array<array<array<f32, kernel_conv_1d_small_kern_input_channels>, kernel_conv_1d_small_kern_nk>, kernel_conv_1d_small_kern_output_channels_per_warp>;
// var<workgroup> workgroup_data_input:  array<array<f32, kernel_conv_1d_small_kern_num_threads_x>, kernel_conv_1d_small_kern_input_channels>;


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

//     var output = array<f32, kernel_conv_1d_small_kern_output_values_per_thread>();

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

//     var output = array<f32, kernel_conv_1d_small_kern_output_values_per_thread>();

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
        output += get_src2(0u, global_id.y, 0u);
    }

    if (has_inject_signal) {
        output += get_src3(u32(tensor_dimension_params.src[3].ne[0]) - output_len + global_id.x, global_id.y, global_id.z);
    }

    let in_idx_offset = input_len - output_len + global_id.x;
    for (var ic = 0u; ic < input_channels; ic = ic + 1u) {
        let input = get_src1(in_idx_offset, ic, global_id.z);
        let kernel = get_src0(global_id.y, ic, 0u);
        output = output + input * kernel;
    }

    set_dst(global_id.x, global_id.y, global_id.z, output);
}


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


@compute
@workgroup_size(256)
fn kernel_scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_el_dst()) {
        return;
    }
    set_dst_lin(global_id.x, get_src0_lin(global_id.x) * get_src1_lin(0u));
}

@compute
@workgroup_size(256)
fn kernel_scale_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_el_dst()) {
        return;
    }
    set_dst_lin(global_id.x, get_dst_lin(global_id.x) * get_src1_lin(0u));
}


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


@compute
@workgroup_size(256)
fn kernel_sqr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_el_dst()) {
        return;
    }
    let x = get_src0_lin(global_id.x);
    set_dst_lin(global_id.x, x * x);
}


@compute
@workgroup_size(256)
fn kernel_sum(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let ne00 = tensor_dimension_params.src[0].nb[1] / tensor_dimension_params.src[0].nb[0];
    let num_el_src0 = ne00 * u32(tensor_dimension_params.src[0].ne[1] * tensor_dimension_params.src[0].ne[2] * tensor_dimension_params.src[0].ne[3]);

    var sum : f32 = 0.0;
    
    for (var i = local_id.x; i < num_el_src0; i = i + 256u) {
        sum = sum + get_src0_lin(i);
    }

    workgroup_data[local_id.x] = sum;
    workgroupBarrier();

    if (0u == local_id.x) {
        sum = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            sum = sum + workgroup_data[i];
        }
        set_dst_lin(0u, sum);
    }
}


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
            output = output + get_src0_lin(base_idx_src0 + isample) * get_src1_lin(base_idx_src1 + isample);
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

        set_dst(global_id.y, global_id.z, wg_id.x, output);
    }
}


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
            output = output + get_src0_lin(base_idx_src0 + isample) * get_src1_lin(base_idx_src1 + isample);
        }
    }

    workgroup_data[local_id.x] = output;
    workgroupBarrier();

    if (0u == local_id.x) {
        output = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            output = output + workgroup_data[i];
        }

        set_dst(wg_id.x, global_id.y, 0u, output);
    }
}


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
        output = output + get_src0_lin(base_idx_src0 + isample) * get_src1_lin(base_idx_src1 + isample);
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

        set_src5_lin(idx_ir + nb1 * global_id.y + nb2 * idx_ic +  nb3 * idx_ik, output);
    }
}

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
        output = output + get_src5_lin(base_idx_src1 + isample);
    }
    set_dst(global_id.y, global_id.z, wg_id.x, output);
}


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
        output = get_src2(global_id.x, global_id.y, global_id.z);
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
                output = output + get_src0_lin(base_idx_src0 + idx_oc) * get_src1_lin(base_idx_src1 + idx_oc * tensor_dimension_params.src[1].nb[1]);
            }
        }
    }

    set_dst(global_id.x, global_id.y, global_id.z, output);
}


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
        output = src2_v4[idx_src2];
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
                let val_src0 = get_src0_lin(base_idx_src0 + idx_oc);
                let idx_src1 = (base_idx_src1 + idx_oc * tensor_dimension_params.src[1].nb[1]) / 4u;
                output = output + val_src0 * mult1 * src1_v4[idx_src1];
            }
        }
    }

    let dst_idx = (global_id.z * tensor_dimension_params.dst.nb[2] +
        global_id.y * tensor_dimension_params.dst.nb[1] +
        mult_idx) / 4u;
    dst_v4[dst_idx] = output;
}


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
    let offset_ne = offset / 4u;
    let nc_limit = nc + offset_ne;
    let nc_after = nc_out - nc_limit;

    if (global_id.x >= nc_out) {
        return;
    }

    var output : f32 = 0.0;
    if (!zero_out_accumulator) {
        output = get_src0(global_id.x, global_id.y, global_id.z);
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
        output = output + get_src1(idx0, global_id.y, global_id.z);
    }

    set_dst(global_id.x, global_id.y, global_id.z, output);
}


@compute
@workgroup_size(256)
fn kernel_add_and_tanh_back(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let mult_idx = global_id.x * 4u;

    if (mult_idx >= num_el_dst()) {
        return;
    }

    let x = src0_v4[global_id.x];
    let y = src1_v4[global_id.x];
    let z = (1.0 - x*x)*y;

    dst_v4[global_id.x] = z;
}


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
            output = output + get_src0_lin(base_idx_src0);
            output = output + get_src0_lin(base_idx_src0+1u);
            output = output + get_src0_lin(base_idx_src0+2u);
            output = output + get_src0_lin(base_idx_src0+3u);
        }
    }

    workgroup_data[local_id.x] = output;
    workgroupBarrier();

    if (0u == local_id.x) {
        output = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            output = output + workgroup_data[i];
        }

        set_dst(0u, wg_id.x, 0u, output);
    }

}


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

        output = output + mult1 * src0_v4[base_idx_src0_base + isample];
    }

    workgroup_data_v4f[local_id.x] = output;
    workgroupBarrier();

    if (0u == local_id.x) {
        output = vec4f();
        for (var i = 0u; i < 256u; i = i + 1u) {
            output = output + workgroup_data_v4f[i];
        }

        set_src5_lin(wg_id.y + wg_id.x * num_batches, output.x+output.y+output.z+output.w);
    }
}


@compute
@workgroup_size(1)
fn kernel_conv_1d_small_kern_back_bias_stage2(@builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let num_batches = u32(tensor_dimension_params.src[0].ne[2]);

    var output : f32 = 0.0;

    let base_idx_src1 = wg_id.x * num_batches;
    for (var isample = 0u; isample < num_batches; isample = isample + 1u) {
        output = output + get_src5_lin(base_idx_src1 + isample);
    }
    set_dst(0u, wg_id.x, 0u, output);
}




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

    var x = get_src0_lin(global_id.x);
    var g = get_src1_lin(global_id.x);
    var m = get_src2_lin(global_id.x);
    var v = get_src3_lin(global_id.x);

    m = m*beta1 +   g*(1.0 - beta1);
    v = v*beta2 + g*g*(1.0 - beta2);
    let mh = m*beta1h;
    let vh = sqrt(v*beta2h) + eps;
    x = x - mh/vh;

    set_src2_lin(global_id.x, m);
    set_src3_lin(global_id.x, v);
    set_dst_lin(global_id.x, x);
}


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

    var x = get_dst_lin(global_id.x);
    var g = get_src1_lin(global_id.x);
    var m = get_src2_lin(global_id.x);
    var v = get_src3_lin(global_id.x);

    m = m*beta1 +   g*(1.0 - beta1);
    v = v*beta2 + g*g*(1.0 - beta2);
    let mh = m*beta1h;
    let vh = sqrt(v*beta2h) + eps;
    x = x - mh/vh;

    set_src2_lin(global_id.x, m);
    set_src3_lin(global_id.x, v);
    set_dst_lin(global_id.x, x);
}