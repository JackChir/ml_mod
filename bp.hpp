#ifndef _BP_HPP_
#define _BP_HPP_

#include <initializer_list>
#include <iomanip>

#include "mat.hpp"
#include "base_function.hpp"
#include "base_logic.hpp"
#include "update_methods.hpp"
#include "weight_initilizer.hpp"
#include "ht_memory.h"

template<typename val_t, int batch_size, template<typename> class update_method_templ, template<typename> class activate_func, typename init_name_t, int i1, int i2, int...is>
struct bp 
{
	mat<i2, i1, val_t> mt_weight;
	mat<i1, batch_size, val_t> mt_in;
	mat<i2, batch_size, val_t> mt_b;
	using next_net_t = bp<val_t, batch_size, update_method_templ, activate_func, init_name_t, i2, is...>;
	next_net_t net_next;
	update_method_templ<mat<i2, i1, val_t>> ad;
	update_method_templ<mat<i2, batch_size, val_t>> adb;
	activate_func<mat<i2, batch_size, val_t>>	act_func;

	using input_type = mat<i1, batch_size, val_t>;									// ��������
	using ret_type = typename next_net_t::ret_type;					// ���緵������
	static constexpr int out_dim = next_net_t::out_dim;

	bp():net_next(), mt_weight(), ad(), adb()
	{
		weight_initilizer<init_name_t>::cal(mt_weight);
	}

	inline auto forward(const mat<i1, batch_size, val_t>& mt_input)
	{
		mt_in = mt_input;
		return net_next.forward(act_func.forward(mt_weight.dot(mt_input) + mt_b));
	}

	inline auto update(const mat<i2, batch_size, val_t>& mt_delta)
	{
		/*����Ȩֵ����*/
		auto mt_desig_origin = act_func.backward();
		auto mt_desig = mt_desig_origin * mt_delta;							// �ش������sigmoid������˵�ֵ
		auto mt_update = mt_desig.dot(mt_in.t());							// ����Ȩֵ�仯����
		auto mt_ret = mt_weight.t().dot(mt_desig);
		mt_weight = ad.update(mt_weight, mt_update);
		mt_b = adb.update(mt_b, mt_desig);									// ����ƫ����
		return mt_ret;
	}

	inline auto backward(const mat<out_dim, batch_size, val_t>& mt_pre_delta)
	{
		auto mt_delta = net_next.backward(mt_pre_delta);
		return update(mt_delta);
	}

	inline ret_type& get_delta()
	{
		return net_next.get_delta();
	}

	void print() const 
	{
		mt_weight.print();
		net_next.print();
	}

	void update_inert() 
	{
		ad.update_inert();
		adb.update_inert();
		net_next.update_inert();
	}
};

template<typename val_t, int batch_size, template<typename> class update_method_templ, template<typename> class activate_func, typename init_name_t, int i1, int i2>
struct bp<val_t, batch_size, update_method_templ, activate_func, init_name_t, i1, i2>
{
	mat<i2, i1, val_t> mt_weight;
	mat<i1, batch_size, val_t> mt_in;
	mat<i2, batch_size, val_t> mt_out;
	mat<i2, batch_size, val_t> mt_b;					// ������
	//mat<i2, batch_size, val_t> mt_delta;
	update_method_templ<mat<i2, i1, val_t>> ad;
	update_method_templ<mat<i2, batch_size, val_t>> adb;
	activate_func<mat<i2, batch_size, val_t>>	act_func;
	
	static constexpr int out_dim = i2;
	using input_type = mat<i1, batch_size, val_t>;									// ��������
	using ret_type = mat<i2, batch_size, val_t>;

	bp() :mt_weight(), ad(), adb()
	{
		weight_initilizer<init_name_t>::cal(mt_weight);
	}

	inline auto forward(const mat<i1, batch_size, val_t>& mt_input)
	{
		mt_in = mt_input;
		mt_out = act_func.forward(mt_weight.dot(mt_input) + mt_b);
		return mt_out;
	}

	inline auto update(const mat<i2, batch_size, val_t>& mt_delta)
	{
		/*����Ȩֵ����*/
		auto mt_desig_origin = act_func.backward();
		auto mt_desig = mt_desig_origin * mt_delta;			// �ش������sigmoid������˵�ֵ
		auto mt_update = mt_desig.dot(mt_in.t());
		auto mt_ret = mt_weight.t().dot(mt_desig);
		mt_weight = ad.update(mt_weight, mt_update);
		mt_b = adb.update(mt_b, mt_desig);
		return mt_ret;
	}

	inline auto backward(const mat<i2, batch_size, val_t>& mt_delta)
	{
		//mt_delta = mt_out - mt_expected
		return update(mt_delta);
	}
#if 0
	inline ret_type& get_delta()
	{
		return mt_delta;
	}
#endif
	void print() const
	{
		mt_weight.print();
	}

	void update_inert()
	{
		ad.update_inert();
		adb.update_inert();
	}
};

template<typename val_t, int batch_size, template<typename> class update_method_templ, template<typename> class activate_func, typename init_name_t, int i1, int i2, int...is>
void write_file(const bp<val_t, batch_size, update_method_templ, activate_func, init_name_t, i1, i2, is...>& b, ht_memory& mry)
{
	write_file(b.mt_weight, mry);
	write_file(b.mt_b, mry);
	if constexpr (0 != sizeof...(is))
	{
		write_file(b.net_next, mry);
	}
}

template<typename val_t, int batch_size, template<typename> class update_method_templ, template<typename> class activate_func, typename init_name_t, int i1, int i2, int...is>
void read_file(ht_memory& mry, bp<val_t, batch_size, update_method_templ, activate_func, init_name_t, i1, i2, is...>& b)
{
	read_file(mry, b.mt_weight);
	read_file(mry, b.mt_b);
	if constexpr (0 != sizeof...(is))
	{
		read_file(mry, b.net_next);
	}
}

#endif
