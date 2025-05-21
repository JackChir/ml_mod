#ifndef _CASCADE_JUDGER_H_
#define _CASCADE_JUDGER_H_
/** 
 * @file cascade_judger.h
 * @brief 级联分类器的判断器
 * @details
 * 使用决策树结合DBN构建级联分类器，DBN分别基于1/5/10分钟RSI、MACD、KDJ、十档盘口、成交量等数据进行训练，生成结果为200维度的一个向量(涨跌10%，精确到0.1%就是200个档位)，用于指明上涨下跌的幅度；
 * 1、收集一段时间内的数据；
 * 2、使用收集到的数据对DBN进行一定次数的训练；
 * 3、使用数据对决策树进行训练；
 */

 #include "mat.hpp"
 #include "dbn_t.hpp"
 #include "decision_tree.hpp"


 template<typename dbn_name, int input_size, int...hidden_layer_size>
 auto get_dbn_ptr()
 {
    using dbn_type = dbn_t<double, input_size, hidden_layer_size..., 200>;
    static dbn_type s_dbn;
    return &s_dbn;
 }

 template<int data_num>
 struct market_data_for_train
 {
    static const int RSI_SIZE = (data_num + data_num/5 + data_num/10)*3;
    static const int MACD_SIZE = (data_num + data_num/5 + data_num/10)*2;
    static const int KDJ_SIZE = (data_num + data_num/5 + data_num/10)*3;
    static const int PRICE_VOLUME_SIZE = 20 * 2 * data_num;
    int label; // 1分钟后的涨跌幅
 };

#endif
