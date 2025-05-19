#include <iostream>
#include "base_function.hpp"

void test_base_ops()
{
    mat<3, 1, double> mt1 = { 1, 2, 3 };
    mat<1, 3, double> mt2 = { 4, 5, 6 };
    mat<3, 3, double> mt3 = mt1.dot(mt2);
    mt3.print();
    mat<3, 3, double> mt4 = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    mat<3, 3, double> mt5 = mt3 - mt4;
    mt5.print();
    mat<3, 3, double> mt6 = mt5 / 2.0;
    mt6.print();
    mat<3, 3, double> mt7 = mt6 * 2.0;
    mt7.print();
    mat<3, 3, double> mt8 = mt7 + 1.0;
    mt8.print();
}

#include "bp.hpp"
#include "activate_function.hpp"

void test_bp()
{
    using net_t = bp<double, 1, nadam, softmax, HeMean, 3, 10>;
    net_t net;
    using input_t = typename net_t::input_type;
    using ret_t = typename net_t::ret_type;
    input_t mt_input = { .1, .2, .3 };
    ret_t mt_expected = { 0, 1., 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < 50000; ++i)
    {
        ret_t mt_out = net.forward(mt_input);
        net.backward(mt_out - mt_expected );
        net.update_inert();
    }
    net.forward(mt_input).print();
}

#include "restricked_boltzman_machine.hpp"

void test_rbm()
{
    using net_t = restricked_boltzman_machine<3, 16>;

    net_t rbm;
    using input_t = mat<3, 1, double>;
    auto train = {
        input_t{ 1, 0, 1 },
        input_t{ 0, 0, 1 },
        input_t{ 1, 0, 0 },
        input_t{ 1, 1, 1 },
        input_t{ 0, 0, 0 }
    };
    // 经过试验1W次训练可以让rbm记住这些数据
    for (int i = 0; i < 10000; ++i)
    {
        for (auto&& mt_input : train)
        {
            rbm.train(mt_input);
        }
    }
    for (auto&& mt_input : train)
    {
        auto mt_out = rbm.association(mt_input*0.8);
        mt_out.print();
    }
}

#include <random>
#include "gmm_t.hpp"

void test_gmm()
{
    // 测试高斯混合模型
    constexpr int dim_size = 2;
	using mat_type = mat<dim_size, 1, double>;
	std::vector<mat_type> vec_x;

	/* 生成数据 */
	std::default_random_engine ge;
	std::normal_distribution <double> ud(0., 5.);
	const int set_size = 10;
	for (int i = 0; i < set_size; ++i)
	{
		mat_type mt;
		for (int i = 0; i < mt.size(); ++i) 
		{
			mt.get(i, 0) = ud(ge);
		}
		vec_x.push_back(mt);
	}
	std::normal_distribution <double> ud1(20., 8.);
	for (int i = 0; i < 2* set_size; ++i)
	{
		mat_type mt;
		for (int i = 0; i < dim_size; ++i)
		{
			mt.get(i, 0) = ud1(ge);
		}
		vec_x.push_back(mt);
	}
	std::normal_distribution <double> ud2(30., 4.);
	for (int i = 0; i < 2* set_size; ++i)
	{
		mat_type mt;
		for (int i = 0; i < dim_size; ++i)
		{
			mt.get(i, 0) = ud2(ge);
		}
		vec_x.push_back(mt);
	}
	/* 初始化类别 */
	std::random_device rd;
	std::mt19937 rng(rd());
	std::shuffle(vec_x.begin(), vec_x.end(), rng);

	std::vector<struct em_class<dim_size>> vec_cls(3);
	class def_init;
	for (int lp = 0; lp < vec_cls.size(); ++lp)
	{
		weight_initilizer<def_init>::cal(vec_cls[lp].u, 0., 50.);           // 随机分配0-50之间的均值
		weight_initilizer<def_init>::cal(vec_cls[lp].sigma, 0., 50.);       // 随机分配0-50之间的方差
		vec_cls[lp].sigma = vec_cls[lp].sigma + vec_cls[lp].sigma.t();      // 保证协方差矩阵是对称的
		vec_cls[lp].p = (1./vec_cls.size());                                // 初始化为相同权重
	}
	/* 执行分类 */
	gmm(vec_x, vec_cls, 5);
	/* 展示分类结果 */
	for (int lp = 0; lp < vec_cls.size(); ++lp) 
	{
		printf("---------- %d -----------\r\n", lp);
		vec_cls[lp].u.print();
		vec_cls[lp].sigma.print();
		printf("p=%lf\r\n", vec_cls[lp].p);
	}
}

int main(int argc, char** argv)
{
    //test_rbm();
    test_gmm();
    return 0;
}