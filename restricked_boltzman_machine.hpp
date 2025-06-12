#ifndef _RESTRICKED_BOLTZMAN_MACHINE_HPP_
#define _RESTRICKED_BOLTZMAN_MACHINE_HPP_
#include "mat.hpp"
#include "base_logic.hpp"
#include "activate_function.hpp"
#include "weight_initilizer.hpp"
#include "update_methods.hpp"

static std::default_random_engine e;
static std::uniform_real_distribution<double> ud(0., 1.);

template<int r, int c>
struct bi_mat_accumulate
{
	template<typename imatt, typename vt = double>
	static vt cal(const imatt& mt, const vt& v_threshold, int* const p_accu_num_out)
	{
		vt v_ret;
		if (mt.get(r,c) > v_threshold)
		{
			v_ret = 1.;
			if (p_accu_num_out)
				(*p_accu_num_out)++;
		}
		else 
		{
			v_ret = 0.;
		}
		return v_ret;
	}
};

template<int r, int c>
struct bi_mat
{
	template<typename imatt, typename vt = double>
	static vt cal(const imatt& mt, const vt& v_threshold)
	{
		vt v_ret;
		if (mt.get(r,c) > v_threshold)
		{
			v_ret = 1.;
		}
		else
		{
			v_ret = 0.;
		}
		return v_ret;
	}
};

template<typename target_t>
target_t bi(const target_t& mt_input) 
{
	target_t mt_output;
	col_loop<target_t::c - 1, bi_mat>(mt_output, mt_input, .5);
	return mt_output;
}

template<int r, int c>
struct n_choice
{
	template<typename imatt, typename vt = double>
	static vt cal(const imatt& mt_ratio)
	{
		auto d_ratio = mt_ratio.get(r,c);
		double d_rand = ud(e);
		//printf("input:%lf, rand:%lf\r\n", d_ratio, d_rand);
		return d_ratio < d_rand ? 0. : 1.;
	}
};

// 从输入矩阵mt_ratio中获取第r行第c列的值，
// 与随机数比较，返回0或1
template<typename imatt, typename vt = double>
vt f_choice(const imatt& mt_ratio, const int r, const int c)
{
	auto d_ratio = mt_ratio.get(r,c);
	double d_rand = ud(e);
	return d_ratio < d_rand ? 0. : 1.;
}

// 对输入矩阵mt_input进行采样，返回一个新的矩阵mt_output
// 采样的方式是对mt_input的每个元素进行随机选择
template<typename target_t>
target_t choice(const target_t& mt_input) 
{
	target_t mt_output;
	//col_loop<target_t::c - 1, n_choice>(mt_output, mt_input);
	for (int i = 0; i < mt_input.r; ++i)
	{
		for (int j = 0; j < mt_input.c; ++j)
		{
			mt_output.get(i, j) = f_choice(mt_input, i, j);
		}
	}
	return mt_output;
}

template<int v_num, int h_num, typename val_t = double, template<typename> class um_tpl = nadam>
struct restricked_boltzman_machine 
{
	mat<v_num, h_num, val_t>	W;			// 权值矩阵
	mat<v_num, 1, val_t>		a;			// 显层偏移
	mat<h_num, 1, val_t>		b;			// 隐层偏移
	//val_t						lr;			// 学习率
	um_tpl<mat<v_num, h_num, val_t> > W_updater;	// 权值更新器
	um_tpl<mat<v_num, 1, val_t> > a_updater;	// 显层偏移更新器
	um_tpl<mat<h_num, 1, val_t> > b_updater;	// 隐层偏移更新器


	template<typename T>
	T prob_func(const T& t_in) 
	{
		T t_out;
		//col_loop<T::c - 1, n_sigmoid>(t_out, t_in);
		for (int i = 0; i < t_in.r; ++i)
		{
			for (int j = 0; j < t_in.c; ++j)
			{
				t_out.get(i, j) = f_sigmoid(t_in.get(i, j));
			}
		}
		return t_out;
	}

	restricked_boltzman_machine():W()
	{
		weight_initilizer<XavierMean>::cal(W);
	}

	// 训练的时候必须采样，而前向输出时候可以不进行采样
	void train(const mat<v_num, 1, val_t>& v_input) 
	{
		auto v1 = v_input;
		auto h1 = choice(prob_func(W.t().dot(v1) + b));
		mat<v_num, 1, val_t> v2 = choice(prob_func(W.dot(h1) + a));
		auto h2 = choice(prob_func(W.t().dot(v2) + b));

		//auto cdw = v1.dot(h1.t()) - v2.dot(h2.t());
		//auto cdv = (v1 - v2);
		//auto cdh = (h1 - h2);
		auto cdw = v2.dot(h2.t()) - v1.dot(h1.t());
		auto cdv = (v2 - v1);
		auto cdh = (h2 - h1);
		W = W_updater.update(W, cdw);
		a = a_updater.update(a, cdv);
		b = b_updater.update(b, cdh);
	}

	// 显层输入，求出隐层输出
	mat<h_num, 1, val_t> forward(const mat<v_num, 1, val_t>& v_in, const bool& sample = true)
	{
		auto ret = prob_func(W.t().dot(v_in) + b);
		if (sample)
		{
			return choice(ret);
		}
		else
		{
			return ret;
		}
	}

	// 隐层输入，求显层输出
	mat<v_num, 1, val_t> backward(const mat<h_num, 1, val_t>& h1, const bool& sample = true)
	{
		auto ret = prob_func(W.dot(h1) + a);
		if (sample)
		{
			return choice(ret);
		}
		else
		{
			return ret;
		}
	}

	// 显层输入，经过隐层进行联想，最后再反馈回显层输出
	mat<v_num, 1, val_t> association(const mat<v_num, 1, val_t>& v_in, const bool& sample = true)
	{
		auto h1 = forward(v_in, sample);
		auto v_out = backward(h1, sample);
		return v_out;
	}

	void update_inert()
	{
		W_updater.update_inert();
		a_updater.update_inert();
		b_updater.update_inert();
	}

	void print()
	{
		printf("###### %10s ######\r\n", "RBM");
		W.print();
		a.print();
		b.print();
		printf("###### %10s ######\r\n", "RBM END");
	}
};

#include "ht_memory.h"

template<int v_num, int h_num, typename val_t = double>
void write_file(const restricked_boltzman_machine<v_num, h_num, val_t>& rbm, ht_memory& mry)
{
	mry.write(rbm.W);
	mry.write(rbm.a);
	mry.write(rbm.b);
}

template<int v_num, int h_num, typename val_t = double>
void read_file(ht_memory& mry, restricked_boltzman_machine<v_num, h_num, val_t>& rbm)
{
	mry.read(rbm.W);
	mry.read(rbm.a);
	mry.read(rbm.b);
}

#endif
