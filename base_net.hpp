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
		// 按照列进行归一化，即求列的均值和方差，然后在每一列上进行归一化处理
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

#endif
