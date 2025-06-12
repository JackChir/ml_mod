#ifndef _DBN_HPP_
#define _DBN_HPP_

#include "mat.hpp"
#include "restricked_boltzman_machine.hpp"
#include "loss_function.hpp"

/*
DBN的主要思路是通过RBM对输入进行编码，然后将编码后的数据通过BP神经网络进行模式判断
*/
template<template<int> class predict_t, typename val_t, int iv, int ih, int...is>
struct dbn_t
{
	restricked_boltzman_machine<iv, ih, val_t, nadam>	rbm;
	dbn_t<predict_t, val_t, ih, is...>						dbn_next;
	using next_type = dbn_t<predict_t, val_t, ih, is...>;
    using input_type = mat<iv, 1, val_t>;
	using ret_type = typename next_type::ret_type;
	using pretrain_ret_type = typename next_type::pretrain_ret_type;


	void pretrain(const std::vector<mat<iv, 1> >& vec, const int& i_epochs = 100, const bool& sample = true) 
	{
		/* 训练当前层 */
		for (int i = 0; i < i_epochs; ++i)
		{
			if (i % 16 == 15)
			{
				rbm.update_inert(); 
			}
			for (auto itr = vec.begin(); itr != vec.end(); ++itr) 
			{
				rbm.train(*itr);
			}
		}
		/* 准备下层数据 */
		std::vector<mat<ih, 1, val_t> > vec_hs;
		for (auto itr = vec.begin(); itr != vec.end(); ++itr)
		{
			vec_hs.push_back(rbm.forward(*itr, sample));
		}
		/* 用隐含层结果训练下一层 */
		dbn_next.pretrain(vec_hs, i_epochs, sample);
	}

	inline std::vector<pretrain_ret_type>& get_pretrain_result()
	{
		return dbn_next.get_pretrain_result();
	}

	template<typename loss_func_t = cross_entropy >
	void finetune(const std::vector<ret_type>& vec_expected, const int& i_epochs = 100)
	{
		dbn_next.template finetune<loss_func_t>(vec_expected, i_epochs);              // 让最后一层bp层进行训练
	}

	auto forward(const mat<iv, 1>& v1, const bool& sample = true)
	{
		return dbn_next.forward(rbm.forward(v1, sample), sample);
	}

};

template<template<int> class predict_t, typename val_t, int iv, int ih>
struct dbn_t<predict_t, val_t, iv, ih> 
{
	restricked_boltzman_machine<iv, ih, val_t, nadam>	rbm;
	//bp<val_t, 1, nadam, softmax, XavierGaussian, ih, ih>	predict_net;						// 最后加上一个softmax作为激活函数的bp神经网络
	predict_t<ih> predict_net;						// 最后加上一个softmax作为激活函数的bp神经网络
	std::vector<mat<ih, 1, val_t> >				vec_pretrain_result;							// 用于暂存pretrain的结果，用于给predict_net进行finetune

	using ret_type = typename predict_t<ih>::ret_type;	// 预测结果类型
	using pretrain_ret_type = mat<ih, 1, val_t>;

	void pretrain(const std::vector<mat<iv, 1> >& vec, const int& i_epochs = 100, const bool& sample = true)
	{
		/* 训练当前层 */
		for (int i = 0; i < i_epochs; ++i)
			for (auto itr = vec.begin(); itr != vec.end(); ++itr)
			{
				rbm.train(*itr);
			}
		vec_pretrain_result.clear();
		for (auto itr = vec.begin(); itr != vec.end(); ++itr)
		{
			vec_pretrain_result.push_back(rbm.forward(*itr, sample));
		}
	}

	inline std::vector<pretrain_ret_type>& get_pretrain_result()
	{
		return vec_pretrain_result;
	}

	template<typename loss_func_t = cross_entropy >
	void finetune(const std::vector<ret_type>& vec_expected, const int& i_epochs = 100)
	{
		for (int i = 0; i < i_epochs; ++i) 
		{
			auto itr_expected = vec_expected.begin();
			auto itr_input = vec_pretrain_result.begin();
			for (; itr_expected != vec_expected.end() && itr_input != vec_pretrain_result.end(); ++itr_expected, ++itr_input)
			{
				auto ret = predict_net.forward(*itr_input);				// 得到bp层的输出
				predict_net.backward(loss_function<loss_func_t>::cal(ret, *itr_expected));	// 得到误差值
			}
		}
		vec_pretrain_result.clear(); // 清空预训练结果
	}

	auto forward(const mat<iv, 1>& v1, const bool& sample = true)
	{
		return predict_net.forward(rbm.forward(v1, sample));
	}
};


template<template<int> class predict_t, typename val_t, int iv, int ih, int...is>
void write_file(const dbn_t<predict_t, val_t, iv, ih, is...>& dbn, ht_memory& mry)
{
	if constexpr (0 != sizeof...(is))
	{
		write_file(dbn.dbn_next, mry);
	}
	if constexpr (0 == sizeof...(is))
	{
		write_file(dbn.predict_net, mry);
	}
}

template<template<int> class predict_t, typename val_t, int iv, int ih, int...is>
void read_file(ht_memory& mry, dbn_t<predict_t, val_t, iv, ih, is...>& dbn)
{
	if constexpr (0 != sizeof...(is))
	{
		read_file(mry, dbn.dbn_next);
	}
	if constexpr (0 == sizeof...(is))
	{
		read_file(mry, dbn.predict_net);
	}
}


#endif
