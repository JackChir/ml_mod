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

int main(int argc, char** argv)
{
    test_bp();
    return 0;
}