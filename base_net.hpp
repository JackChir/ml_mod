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
};

#endif
