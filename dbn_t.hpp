#ifndef _DBN_HPP_
#define _DBN_HPP_

#include "bp.hpp"
#include "restricked_boltzman_machine.hpp"

/*
DBN的主要思路是通过RBM对输入进行编码，然后将编码后的数据通过BP神经网络进行模式判断
*/

template<typename val_t, int iv, int ih, int...is>
struct dbn_t
{
	restricked_boltzman_machine<iv, ih, val_t>	rbm;
	dbn_t<val_t, ih, is...>						dbn_next;
	using ret_type = typename dbn_t<val_t, ih, is...>::ret_type;


	void pretrain(const std::vector<mat<iv, 1> >& vec, const int& i_epochs = 100) 
	{
		/* 训练当前层 */
		for (int i = 0; i < i_epochs; ++i)
			for (auto itr = vec.begin(); itr != vec.end(); ++itr) 
			{
				rbm.train(*itr);
			}
		/* 准备下层数据 */
		std::vector<mat<ih, 1, val_t> > vec_hs;
		for (auto itr = vec.begin(); itr != vec.end(); ++itr)
		{
			vec_hs.push_back(rbm.forward(*itr));
		}
		/* 用隐含层结果训练下一层 */
		dbn_next.pretrain(vec_hs, i_epochs);
	}

	void finetune(const std::vector<ret_type>& vec_expected, const int& i_epochs = 100)
	{
		dbn_next.finetune(vec_expected, i_epochs);              // 让最后一层bp层进行训练
	}

	auto forward(const mat<iv, 1>& v1)
	{
		return dbn_next.forward(rbm.forward(v1));
	}
};

template<typename val_t, int iv, int ih>
struct dbn_t<val_t, iv, ih> 
{
	restricked_boltzman_machine<iv, ih, val_t>	rbm;
	bp<val_t, 1, nadam, softmax, XavierGaussian, ih, ih>	bp_net;						// 最后加上一个softmax作为激活函数的bp神经网络
	std::vector<mat<ih, 1, val_t> >				vec_pretrain_result;

	using ret_type = mat<ih, 1, val_t>;

	void pretrain(const std::vector<mat<iv, 1> >& vec, const int& i_epochs = 100)
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
			vec_pretrain_result.push_back(rbm.forward(*itr));
		}
	}

	void finetune(const std::vector<ret_type>& vec_expected, const int& i_epochs = 100)
	{
		for (int i = 0; i < i_epochs; ++i) 
		{
			auto itr_expected = vec_expected.begin();
			auto itr_input = vec_pretrain_result.begin();
			for (; itr_expected != vec_expected.end() && itr_input != vec_pretrain_result.end(); ++itr_expected, ++itr_input)
			{
				auto ret = bp_net.forward(*itr_input);				// 得到bp层的输出
				bp_net.backward(ret - *itr_expected);				// 得到误差值
			}
		}
	}

	auto forward(const mat<iv, 1>& v1)
	{
		return bp_net.forward(rbm.forward(v1));
	}
};

#endif
