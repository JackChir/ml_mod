#ifndef _ACTIVATE_FUNCTION_HPP_
#define _ACTIVATE_FUNCTION_HPP_
#include <math.h>
#include "base_logic.hpp"
#include "mat.hpp"

template<typename val_t = double>
val_t f_sigmoid(const val_t& v)
{
	return 1. / (1. + exp(-1. * v));
}

template<int r, int c>
class n_sigmoid
{
public:
	template<typename imatt>
	static typename imatt::type cal(const imatt& mt)
	{
		return f_sigmoid(mt.get(r,c));
	}
};

template<typename target_t>
inline target_t sigmoidm(const target_t& mt_input) 
{
	target_t mt_ret;
	col_loop<target_t::c - 1, n_sigmoid>(mt_ret, mt_input);
	return mt_ret;
}

template<typename target_t>
struct sigmoid 
{
	target_t mt_pre_output;
	inline target_t forward(const target_t& mt_input)
	{
		col_loop<target_t::c - 1, n_sigmoid>(mt_pre_output, mt_input);
		return mt_pre_output;
	}

	inline target_t backward()
	{
		typename target_t::type one(1.);
		return mt_pre_output * (one - mt_pre_output);
	}
};

template<>
struct sigmoid<double>
{
	double mt_pre_output;
	inline double forward(const double& mt_input)
	{
		//col_loop<target_t::c - 1, n_sigmoid>(mt_pre_output, mt_input);
		mt_pre_output = f_sigmoid(mt_input);
		return mt_pre_output;
	}

	inline double backward()
	{
		return mt_pre_output * (1. - mt_pre_output);
	}
};

template<int r, int c>
class n_ReLu
{
public:
	template<typename imatt>
	static typename imatt::type cal(const imatt& mt)
	{
		using type = typename imatt::type;
		type v = mt.get(r, c);
		return max_and_choose(v, type(0), type(0.), v);
//		return v < 0 ? 0 :v;
	}
};

template<int r, int c>
class n_ReLu_back
{
public:
	template<typename imatt>
	static typename imatt::type cal(const imatt& mt)
	{
		using type = typename imatt::type;
		type v = mt.get(r,c);
		return max_and_choose(v, type(0.), type(0.), type(1.));
		//return v <= 0. ? 0. : 1.;
	}
};

template<typename target_t>
struct ReLu 
{
	target_t mt_pre_input;
	inline target_t forward(const target_t& mt_input)
	{
		mt_pre_input = mt_input;
		target_t mt_output;
		col_loop<target_t::c - 1, n_ReLu>(mt_output, mt_input);
		return mt_output;
	}

	inline target_t backward()
	{
		target_t mt_output;
		col_loop<target_t::c - 1, n_ReLu_back>(mt_output, mt_pre_input);
		return mt_output;
	}
};

template<>
struct ReLu<double>
{
	double mt_pre_input;
	inline double forward(const double& mt_input)
	{
		mt_pre_input = mt_input;
		return mt_pre_input < 0 ? 0 : mt_pre_input;
	}

	inline double backward()
	{
		return mt_pre_input < 0. ? 0. : 1.;
	}
};

template<typename target_t>
struct softmax 
{
	target_t mt_pre_output;
	inline target_t forward(const target_t& mt_input)
	{
		/* ������е�����exp�� */
		using val_t = typename target_t::type;
		val_t d_max = mt_input.max();
		auto mt1 = mt_input - d_max;
		auto mt_exp = exp(mt1);
		val_t d_sum = mt_exp.sum();
		mt_pre_output = mt_exp / d_sum;
		return mt_pre_output;
	}

	inline target_t backward() 
	{
		using val_t = typename target_t::type;
		val_t one(1.);
		return mt_pre_output * (one - mt_pre_output);
		//return one;				// 使用交叉熵损失函数时，softmax的反向传播不需要乘以(1 - softmax)
	}
};


template<typename target_t>
struct no_activate
{
	target_t mt_pre_input;
	inline target_t forward(const target_t& mt_input)
	{
		return mt_input;
	}

	inline target_t backward()
	{
		return target_t(1.);
	}
};

template<>
struct no_activate<double>
{
	double mt_pre_input;
	inline double forward(const double& mt_input)
	{
		return mt_input;
	}

	inline double backward()
	{
		return 1.;
	}
};

#endif
