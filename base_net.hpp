#ifndef __BASE_NET_HPP__
#define __BASE_NET_HPP__
#include "mat.hpp"

template<typename target_t>
struct normalize_layer_t
{
	target_t mt_pre_input;
    target_t mt_pre_output;
	target_t mt_mean;
	target_t mt_sqrt;
	inline target_t forward(const target_t& mt_input)
	{
		mt_pre_input = mt_input;
		// 按照列进行归一化，即求列的均值和方差，然后在每一列上进行归一化处理，由于每一列代表一个token，所以这种方法被称为layer normalization
		// 与之相对的是batch normalization，它是对每一行进行归一化处理，即求行的均值和方差。
        mt_pre_output = normalize(mt_pre_input, mt_mean, mt_sqrt);
		return mt_pre_output;
	}

	inline target_t backward(const target_t& delta)
	{
        mat<1, target_t::r, typename target_t::type> mt_one(1.0);
        mat<target_t::r, 1, typename target_t::type> mt_one_col(1.0);
        auto S1 = mt_one_col.dot(mt_one.dot(delta));
        auto S2 = mt_one_col.dot(mt_one.dot(delta * mt_pre_output));
        typename target_t::type r = static_cast<typename target_t::type>(target_t::r);
        return (delta - (S1 + S2 * mt_pre_output)/r) / mt_sqrt;
	}
};

// 带有残差链接的网络，模板参数为网络类型
template<typename net_t>
struct residual_layer_t
{
    net_t net;  // 网络
    normalize_layer_t<typename net_t::input_type> norm_layer;  // 归一化层

	// 前向传播时候会自动先进行残差连接，然后再进行层归一化(layer normalization，注意，不是batch normalization，默认数据是列向量)
    typename net_t::ret_type forward(const typename net_t::input_type& input)
    {
        auto output = net.forward(input);  // 前向传播
        return norm_layer.forward(output + input);  // 返回归一化后的输出
    }

    typename net_t::input_type backward(const typename net_t::input_type& delta)
    {
        return net.backward(norm_layer.backward(delta)) + delta;  // 反向传播
    }

    typename net_t::ret_type forward(const typename net_t::encoder_input_type& encoder_input, const typename net_t::decoder_input_type& decoder_input)
    {
        auto output = net.forward(encoder_input, decoder_input);  // 前向传播
        return norm_layer.forward(output + encoder_input);  // 返回归一化后的输出
    }

    void backward(const typename net_t::ret_type& delta, typename net_t::encoder_input_type& encoder_delta, typename net_t::decoder_input_type& decoder_delta)
    {
        auto delta_out = norm_layer.backward(delta);  // 反向传播
        net.backward(delta_out, encoder_delta, decoder_delta);  // 返回误差
    }
};

template<typename net1_t, template<int> class net2_t>
struct join_net
{
    using input_type = typename net1_t::input_type;
    using ret_type = typename net2_t<net1_t::ret_type::r>::ret_type;
    net1_t net1;  // 第一个网络
    net2_t<net1_t::ret_type::r> net2;  // 第二个网络，模板参数为第一个网络的输出维度
    // 前向传播
    ret_type forward(const input_type& input)
    {
        auto output1 = net1.forward(input);  // 第一个网络的输出
        return net2.forward(output1);  // 第二个网络的输出
    }
    // 反向传播
    input_type backward(const ret_type& delta)
    {
        auto delta1 = net2.backward(delta);  // 第二个网络的误差
        return net1.backward(delta1);  // 第一个网络的误差
    }
    void update_inert()
    {
        net1.update_inert();  // 更新第一个网络的参数
        net2.update_inert();  // 更新第二个网络的参数
    }
};

// 旋转位置编码
template<int input_size>
struct RoPEPrecompute
{
    mat<24*60, input_size / 2, double> cos_theta; // 预计算的cos值
    mat<24*60, input_size / 2, double> sin_theta; // 预计算的sin值
    RoPEPrecompute()
    {
        //printf("RoPEPrecompute: input_size=%d theta.size=[%d,%d]*2\r\n", input_size, 24*60, input_size / 2);
        for (int m = 0; m < 24 * 60; ++m)
        {
            for (int k = 0; k < input_size / 2; ++k)
            {
                double theta = m * pow(10000, -2 * k / (input_size / 2));
                cos_theta.get(m, k) = cos(theta);
                sin_theta.get(m, k) = sin(theta);
            }
        }
        //printf("RoPEPrecompute initialized.\r\n");
    }
    void apply(mat<input_size, 1, double>& mt_input, int d_time) const
    {
        for (int i = 0; i < input_size / 2; ++i)
        {
            double cos_val = cos_theta.get(d_time, i);
            double sin_val = sin_theta.get(d_time, i);
            double d1 = mt_input.get(i*2, 0);
            double d2 = mt_input.get(i*2 + 1, 0);
            mt_input.get(i*2, 0) = d1 * cos_val - d2 * sin_val;
            mt_input.get(i*2 + 1, 0) = d1 * sin_val + d2 * cos_val;
        }
    }

    template<int col_num>
    void apply(mat<input_size, col_num, double>& mt_input, mat<col_num, 1, int>& mt_time) const
    {
        for (int i = 0; i < input_size / 2; ++i)
        {
            double cos_val = cos_theta.get(mt_time.get(i, 0), i);
            double sin_val = sin_theta.get(mt_time.get(i, 0), i);
            for (int j = 0; j < col_num; ++j)
            {
                double d1 = mt_input.get(i * 2, j);
                double d2 = mt_input.get(i * 2 + 1, j);
                mt_input.get(i * 2, j) = d1 * cos_val - d2 * sin_val;
                mt_input.get(i * 2 + 1, j) = d1 * sin_val + d2 * cos_val;
            }
        }
    }

    void apply_to_col(mat<input_size, 1, double>& mt_input, int col_idx, int d_time) const
    {
        for (int i = 0; i < input_size / 2; ++i)
        {
            double cos_val = cos_theta.get(d_time, i);
            double sin_val = sin_theta.get(d_time, i);
            double d1 = mt_input.get(i * 2, col_idx);
            double d2 = mt_input.get(i * 2 + 1, col_idx);
            mt_input.get(i * 2, col_idx) = d1 * cos_val - d2 * sin_val;
            mt_input.get(i * 2 + 1, col_idx) = d1 * sin_val + d2 * cos_val;
        }
    }
};

template<int input_size>
RoPEPrecompute<input_size>& get_rope_precompute()
{
    static RoPEPrecompute<input_size> s;
    return s;
}
#endif
