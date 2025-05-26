#ifndef _RESTRICKED_BOLTZMAN_MACHINE_HPP_
#define _RESTRICKED_BOLTZMAN_MACHINE_HPP_
#include "mat.hpp"
#include "base_logic.hpp"
#include "activate_function.hpp"
#include "weight_initilizer.hpp"

static std::default_random_engine e;
static std::uniform_real_distribution<double> ud(0., 1.);

template<int r, int c>
struct bi_mat_accumulate
{
	template<typename imatt, typename vt = double>
	static vt cal(const imatt& mt, const vt& v_threshold, int* const p_accu_num_out)
	{
		vt v_ret;
		if (mt.template get_val<r, c>() > v_threshold)
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
		if (mt.template get_val<r, c>() > v_threshold)
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
		auto d_ratio = mt_ratio.template get_val<r, c>();
		double d_rand = ud(e);
		//printf("input:%lf, rand:%lf\r\n", d_ratio, d_rand);
		return d_ratio < d_rand ? 0. : 1.;
	}
};

template<typename target_t>
target_t choice(const target_t& mt_input) 
{
	target_t mt_output;
	col_loop<target_t::c - 1, n_choice>(mt_output, mt_input);
	return mt_output;
}

template<int v_num, int h_num, typename val_t = double>
struct restricked_boltzman_machine 
{
	mat<v_num, h_num, val_t>	W;			// 权值矩阵
	mat<v_num, 1, val_t>		a;			// 显层偏移
	mat<h_num, 1, val_t>		b;			// 隐层偏移
	val_t						lr;			// 学习率

	template<typename T>
	T prob_func(const T& t_in) 
	{
		T t_out;
		col_loop<T::c - 1, n_sigmoid>(t_out, t_in);
		return t_out;
	}

	restricked_boltzman_machine():W(), lr(0.01)
	{
		weight_initilizer<XavierMean>::cal(W);
	}

	void train(const mat<v_num, 1, val_t>& v_input) 
	{
		auto v1 = v_input;
		auto h1 = choice(prob_func(W.t().dot(v1) + b));
		mat<v_num, 1, val_t> v2 = choice(prob_func(W.dot(h1) + a));
		auto h2 = choice(prob_func(W.t().dot(v2) + b));

		auto cdw = v1.dot(h1.t()) - v2.dot(h2.t());
		auto cdv = (v1 - v2);
		auto cdh = (h1 - h2);

		W = W + lr * cdw;
		a = a + lr * cdv;
		b = b + lr * cdh;
	}

	// 显层输入，求出隐层输出
	mat<h_num, 1, val_t> forward(const mat<v_num, 1, val_t>& v_in)
	{
		return choice(prob_func(W.t().dot(v_in) + b));
	}

	// 隐层输入，求显层输出
	mat<v_num, 1, val_t> backward(const mat<h_num, 1, val_t>& h1)
	{
		return choice(prob_func(W.dot(h1) + a));
	}

	// 显层输入，经过隐层进行联想，最后再反馈回显层输出
	mat<v_num, 1, val_t> association(const mat<v_num, 1, val_t>& v_in)
	{
		return backward(forward(v_in));
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
	mry.write(rbm.lr);
}

template<int v_num, int h_num, typename val_t = double>
void read_file(ht_memory& mry, restricked_boltzman_machine<v_num, h_num, val_t>& rbm)
{
	mry.read(rbm.W);
	mry.read(rbm.a);
	mry.read(rbm.b);
	mry.read(rbm.lr);
}

#endif
