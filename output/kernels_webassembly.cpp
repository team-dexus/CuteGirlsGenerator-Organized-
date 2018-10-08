
#include <stdlib.h>
#include <math.h>

float static_buffer[70501334];
float* dynamic_buffer = nullptr;

int meta_buf_0[] = {990150,990278,128};
int meta_buf_1[] = {990278,990406,128,1023410176};
int meta_buf_2[] = {990406,990534,1,128,1,1,4,4,4,4,1,1,1,1,3,3};
int meta_buf_3[] = {990534,0,1023302,16,64,2048};
int meta_buf_4[] = {131072,1023302,1024326,64,64,64,16,1045220557,1030839279};
int meta_buf_5[] = {1024326,1025350,1,64,4,4,4,4,3,3,1,1,1,1,1,1};
int meta_buf_6[] = {1025350,131136,1034566,16,64,576};
int meta_buf_7[] = {168000,1034566,1035590,64,16,64,16};
int meta_buf_8[] = {1035590,1036614,1024,1073741824};
int meta_buf_9[] = {1036614,1037638,16,16,4,1,1,1,4,4,1024,16,4,1,4,64,16,16};
int meta_buf_10[] = {1035590,1037638,1037654,16,64,16,64,1015021568,841731191,1056964608,1045220557};
int meta_buf_11[] = {1037654,1038678,1,4,4,64,8,8,2,2,2,2,0,0};
int meta_buf_12[] = {1038678,1042774,4096,1030839279};
int meta_buf_13[] = {1042774,1046870,1,64,8,8,8,8,3,3,1,1,1,1,1,1};
int meta_buf_14[] = {1046870,168064,1083734,64,64,576};
int meta_buf_15[] = {204928,1083734,1087830,64,64,64,64};
int meta_buf_16[] = {1087830,1091926,4096,1073741824};
int meta_buf_17[] = {1091926,1096022,64,64,8,1,1,1,8,8,4096,64,8,1,4,64,64,64};
int meta_buf_18[] = {1087830,1096022,1096086,64,64,64,64,1015021568,841731191,1056964608,1045220557,1030839279};
int meta_buf_19[] = {1096086,1100182,1,64,8,8,8,8,3,3,1,1,1,1,1,1};
int meta_buf_20[] = {1100182,204992,1137046,64,64,576};
int meta_buf_21[] = {241856,1137046,1141142,64,64,64,64};
int meta_buf_22[] = {1141142,1145238,4096,1073741824};
int meta_buf_23[] = {1145238,1149334,64,64,8,1,1,1,8,8,4096,64,8,1,4,64,64,64};
int meta_buf_24[] = {1149334,1141142,1149398,64,64,64,64,1015021568,841731191,1056964608,1045220557};
int meta_buf_25[] = {1149398,1153494,1,8,8,64,16,16,2,2,2,2,0,0};
int meta_buf_26[] = {1153494,1169878,16384,1030839279};
int meta_buf_27[] = {1169878,1186262,1,64,16,16,16,16,3,3,1,1,1,1,1,1};
int meta_buf_28[] = {1186262,241920,1333718,256,64,576};
int meta_buf_29[] = {278784,1333718,1350102,64,256,64,256};
int meta_buf_30[] = {1350102,1366486,16384,1073741824};
int meta_buf_31[] = {1366486,1382870,256,256,16,1,1,1,16,16,16384,256,16,1,4,64,256,256};
int meta_buf_32[] = {1382870,1350102,1383126,256,64,256,64,1015021568,841731191,1056964608,1045220557,1030839279};
int meta_buf_33[] = {1383126,1399510,1,64,16,16,16,16,3,3,1,1,1,1,1,1};
int meta_buf_34[] = {1399510,278848,1546966,256,64,576};
int meta_buf_35[] = {315712,1546966,1563350,64,256,64,256};
int meta_buf_36[] = {1563350,1579734,16384,1073741824};
int meta_buf_37[] = {1579734,1596118,256,256,16,1,1,1,16,16,16384,256,16,1,4,64,256,256};
int meta_buf_38[] = {1596118,1563350,1596374,256,64,256,64,1015021568,841731191,1056964608,1045220557};
int meta_buf_39[] = {1596374,1612758,1,16,16,64,32,32,2,2,2,2,0,0};
int meta_buf_40[] = {1612758,1678294,65536,1030839279};
int meta_buf_41[] = {1678294,1743830,1,64,32,32,32,32,3,3,1,1,1,1,1,1};
int meta_buf_42[] = {1743830,315776,2333654,1024,64,576};
int meta_buf_43[] = {352640,2333654,2399190,64,1024,64,1024};
int meta_buf_44[] = {2399190,2464726,65536,1073741824};
int meta_buf_45[] = {2464726,2530262,1024,1024,32,1,1,1,32,32,65536,1024,32,1,4,64,1024,1024};
int meta_buf_46[] = {2530262,2399190,2531286,1024,64,1024,64,1015021568,841731191,1056964608,1045220557,1030839279};
int meta_buf_47[] = {2531286,2596822,1,64,32,32,32,32,3,3,1,1,1,1,1,1};
int meta_buf_48[] = {2596822,352704,3186646,1024,64,576};
int meta_buf_49[] = {389568,3186646,3252182,64,1024,64,1024};
int meta_buf_50[] = {3252182,3317718,65536,1073741824};
int meta_buf_51[] = {3317718,3383254,1024,1024,32,1,1,1,32,32,65536,1024,32,1,4,64,1024,1024};
int meta_buf_52[] = {3383254,3252182,3384278,1024,64,1024,64,1015021568,841731191,1056964608,1045220557};
int meta_buf_53[] = {3384278,3449814,1,32,32,64,64,64,2,2,2,2,0,0};
int meta_buf_54[] = {3449814,3711958,262144,1030839279};
int meta_buf_55[] = {3711958,3974102,1,64,64,64,64,64,3,3,1,1,1,1,1,1};
int meta_buf_56[] = {3974102,389632,6333398,4096,64,576};
int meta_buf_57[] = {426496,6333398,6595542,64,4096,64,4096};
int meta_buf_58[] = {6595542,6857686,262144,1073741824};
int meta_buf_59[] = {6857686,7119830,4096,4096,64,1,1,1,64,64,262144,4096,64,1,4,64,4096,4096};
int meta_buf_60[] = {7119830,6595542,7123926,4096,64,4096,64,1015021568,841731191,1056964608,1045220557,1030839279};
int meta_buf_61[] = {7123926,7386070,1,64,64,64,64,64,3,3,1,1,1,1,1,1};
int meta_buf_62[] = {7386070,426560,9745366,4096,64,576};
int meta_buf_63[] = {463424,9745366,10007510,64,4096,64,4096};
int meta_buf_64[] = {10007510,10269654,262144,1073741824};
int meta_buf_65[] = {10269654,10531798,4096,4096,64,1,1,1,64,64,262144,4096,64,1,4,64,4096,4096};
int meta_buf_66[] = {10531798,10007510,10535894,4096,64,4096,64,1015021568,841731191,1056964608,1045220557};
int meta_buf_67[] = {10535894,10798038,1,64,64,64,128,128,2,2,2,2,0,0};
int meta_buf_68[] = {10798038,11846614,1048576,1030839279};
int meta_buf_69[] = {11846614,12895190,1,64,128,128,128,128,3,3,1,1,1,1,1,1};
int meta_buf_70[] = {12895190,463488,22332374,16384,64,576};
int meta_buf_71[] = {500352,22332374,23380950,64,16384,64,16384};
int meta_buf_72[] = {23380950,24429526,1048576,1073741824};
int meta_buf_73[] = {24429526,25478102,16384,16384,128,1,1,1,128,128,1048576,16384,128,1,4,64,16384,16384};
int meta_buf_74[] = {25478102,23380950,25494486,16384,64,16384,64,1015021568,841731191,1056964608,1045220557,1030839279};
int meta_buf_75[] = {25494486,26543062,1,64,128,128,128,128,3,3,1,1,1,1,1,1};
int meta_buf_76[] = {26543062,500416,35980246,16384,64,576};
int meta_buf_77[] = {537280,35980246,37028822,64,16384,64,16384};
int meta_buf_78[] = {37028822,38077398,1048576,1073741824};
int meta_buf_79[] = {38077398,39125974,16384,16384,128,1,1,1,128,128,1048576,16384,128,1,4,64,16384,16384};
int meta_buf_80[] = {39125974,37028822,39142358,16384,64,16384,64,1015021568,841731191,1056964608,1045220557,1043662067};
int meta_buf_81[] = {39142358,537344,40190934,16384,3,64};
int meta_buf_82[] = {586688,537536,40190934,40240086,16384,3,3,3,16384,-1082130432,1};
int meta_buf_83[] = {40240086,40289238,1,128,128,3,256,256,2,2,2,2,0,0};
int meta_buf_84[] = {40289238,40485846,3,65536,65536,3};
int meta_buf_85[] = {40485846,40682454,65536,3,3,65536,1049319752};
int meta_buf_86[] = {40682454,40879062,1,3,256,256,256,256,3,3,1,1,1,1,1,1};
int meta_buf_87[] = {40879062,586691,42648534,65536,32,27};
int meta_buf_88[] = {587555,42648534,44745686,32,32,32,65536,1045220557,1034594987};
int meta_buf_89[] = {44745686,46842838,1,32,256,256,256,256,3,3,1,1,1,1,1,1};
int meta_buf_90[] = {46842838,587587,65717206,65536,32,288};
int meta_buf_91[] = {596803,65717206,67814358,32,32,32,65536,1045220557,1048576000};
int meta_buf_92[] = {67814358,596835,69911510,65536,3,32};
int meta_buf_93[] = {793539,596931,793542,40485846,69911510,70108118,65536,65536,65536,3,3,3,65536,-1082130432,1,-1082130432,1};
int meta_buf_94[] = {70108118,70304726,1,256,256,3,256,128,1,2,1,2,0,0};
int meta_buf_95[] = {70304726,70403030,3,32768,32768,3};
int* meta_buffers[] = {meta_buf_0,meta_buf_1,meta_buf_2,meta_buf_3,meta_buf_4,meta_buf_5,meta_buf_6,meta_buf_7,meta_buf_8,meta_buf_9,meta_buf_10,meta_buf_11,meta_buf_12,meta_buf_13,meta_buf_14,meta_buf_15,meta_buf_16,meta_buf_17,meta_buf_18,meta_buf_19,meta_buf_20,meta_buf_21,meta_buf_22,meta_buf_23,meta_buf_24,meta_buf_25,meta_buf_26,meta_buf_27,meta_buf_28,meta_buf_29,meta_buf_30,meta_buf_31,meta_buf_32,meta_buf_33,meta_buf_34,meta_buf_35,meta_buf_36,meta_buf_37,meta_buf_38,meta_buf_39,meta_buf_40,meta_buf_41,meta_buf_42,meta_buf_43,meta_buf_44,meta_buf_45,meta_buf_46,meta_buf_47,meta_buf_48,meta_buf_49,meta_buf_50,meta_buf_51,meta_buf_52,meta_buf_53,meta_buf_54,meta_buf_55,meta_buf_56,meta_buf_57,meta_buf_58,meta_buf_59,meta_buf_60,meta_buf_61,meta_buf_62,meta_buf_63,meta_buf_64,meta_buf_65,meta_buf_66,meta_buf_67,meta_buf_68,meta_buf_69,meta_buf_70,meta_buf_71,meta_buf_72,meta_buf_73,meta_buf_74,meta_buf_75,meta_buf_76,meta_buf_77,meta_buf_78,meta_buf_79,meta_buf_80,meta_buf_81,meta_buf_82,meta_buf_83,meta_buf_84,meta_buf_85,meta_buf_86,meta_buf_87,meta_buf_88,meta_buf_89,meta_buf_90,meta_buf_91,meta_buf_92,meta_buf_93,meta_buf_94,meta_buf_95};

extern "C" void init() {
    //static_buffer = (float*)malloc(70501334 * sizeof(float));
}

extern "C" float* get_static_buffer(void) {
    return static_buffer;
}

extern "C" float* allocate_dynamic_buffer(int count) {
    if (dynamic_buffer) {
        free(dynamic_buffer);
        dynamic_buffer = nullptr;
    }
    dynamic_buffer = (float*)malloc(count * sizeof(float));
    return dynamic_buffer;
}

extern "C" float* get_dynamic_buffer(void) {
    return dynamic_buffer;
}
extern "C" void set_placeholder_value(int kernel_order, int offset, int value) {
    meta_buffers[kernel_order][offset] = value;
}

void reshape_f6c88dc1fd9479f62d789912530e6a5c7c51c73de368540e8b8e1ceb(const int * meta_buffer )
{
    const float *x = (static_buffer + meta_buffer[0]);
    float *y = (static_buffer + meta_buffer[1]);

    const int N = meta_buffer[2];

    for (int gid = 0; gid < N; gid += 1) {
        y[gid] = x[gid];
    }
}


void scalarmul_021226a8fb40f401fcd74ecc98cab18a4513f970372147bec0077349(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    float * v2 = (static_buffer + meta_buffer[1]);
    const int D0 = meta_buffer[2];
    int d0;
    for (d0 = 0; d0 < D0; d0 += 1) {
        const float v3 = v1[d0];
        float v4;
        {
            float value = *((float *)(&meta_buffer[3]));
            v4 = v3 * value;
        }
        v2[d0] = v4;
    }
}


void im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(const int * meta_buffer)
{
    const float *im = (static_buffer + meta_buffer[0]);
    float *col = (static_buffer + meta_buffer[1]);

    const int N = meta_buffer[2];
    const int C1 = meta_buffer[3];
    const int H1 = meta_buffer[4];
    const int W1 = meta_buffer[5];
    const int H2 = meta_buffer[6];
    const int W2 = meta_buffer[7];
    const int KH = meta_buffer[8];
    const int KW = meta_buffer[9];
    const int DH = meta_buffer[10];
    const int DW = meta_buffer[11];
    const int SH = meta_buffer[12];
    const int SW = meta_buffer[13];
    const int PH = meta_buffer[14];
    const int PW = meta_buffer[15];

    for (int gid = 0; gid < N*H2*W2*KH*KW*C1; gid += 1) {
        const int c1 = gid % C1;
        const int kw = gid / C1 % KW;
        const int kh = gid / C1 / KW % KH;
        const int w2 = gid / C1 / KW / KH % W2;
        const int h2 = gid / C1 / KW / KH / W2 % H2;
        const int  n = gid / C1 / KW / KH / W2 / H2;

        const int h1 = h2 * SH - PH + kh * DH;
        const int w1 = w2 * SW - PW + kw * DW;

        col[gid] = (h1 < 0 || h1 >= H1 || w1 < 0 || w1 >= W1) ? 0 : im[((n*H1+h1)*W1+w1)*C1+c1];
    }
}


#ifndef INCLUDE_EIGEN
#define INCLUDE_EIGEN
#include <Eigen/Dense>
#endif

void tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(const int * meta_buffer)
{
    float *A = (static_buffer + meta_buffer[0]);
    float *B = (static_buffer + meta_buffer[1]);
    float *C = (static_buffer + meta_buffer[2]);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > a_mat(A, meta_buffer[3], meta_buffer[5]);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > b_mat(B, meta_buffer[5], meta_buffer[4]);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > c_mat(C, meta_buffer[3], meta_buffer[4]);

    c_mat.noalias() = a_mat * b_mat;
}


void fusedelementwise_99aebe006807faa2c23dcb393bb85a063a705d397c501de9b47249da(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    float * v3 = (static_buffer + meta_buffer[2]);
    const int v4 = meta_buffer[3];
    const int v5 = meta_buffer[4];
    const int D0 = meta_buffer[5];
    const int D1 = meta_buffer[6];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        const float v6 = v1[d0];
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            const float v7 = v2[d0 + d1*v4];
            float v8;
            {
                v8 = v7 + v6;
            }
            float v9;
            {
                float slope = *((float *)(&meta_buffer[7]));
                v9 = v8 > 0 ? v8 : (v8 * slope);
            }
            float v10;
            {
                float value = *((float *)(&meta_buffer[8]));
                v10 = v9 * value;
            }
            v3[d0 + d1*v5] = v10;
        }
    }
}


void elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    float * v3 = (static_buffer + meta_buffer[2]);
    const int v4 = meta_buffer[3];
    const int v5 = meta_buffer[4];
    const int D0 = meta_buffer[5];
    const int D1 = meta_buffer[6];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        const float v6 = v1[d0];
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            const float v7 = v2[d0 + d1*v4];
            float v8;
            {
                v8 = v7 + v6;
            }
            v3[d0*v5 + d1] = v8;
        }
    }
}


void scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    float * v2 = (static_buffer + meta_buffer[1]);
    const int D0 = meta_buffer[2];
    int d0;
    for (d0 = 0; d0 < D0; d0 += 1) {
        const float v3 = v1[d0];
        float v4;
        {
            float value = *((float *)(&meta_buffer[3]));
            v4 = powf(v3, value);
        }
        v2[d0] = v4;
    }
}


void sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(const int * meta_buffer)
{
    const float *X = (static_buffer + meta_buffer[0]);
    float *Y = (static_buffer + meta_buffer[1]);
    const int *y_stride = (&(meta_buffer[2]));
    const int *y_shape = (&(meta_buffer[6]));
    const int *x_stride = (&(meta_buffer[10]));
    const int D = meta_buffer[14];
    const int N = meta_buffer[15];
    const int MAX_GID = meta_buffer[16];
    const int x_target_axis_stride = meta_buffer[17];

    for (int gid = 0; gid < MAX_GID; gid++) {
        int x_index = 0;
        for (int d = 0; d < D; d++) x_index += ((gid / y_stride[d]) % y_shape[d]) * x_stride[d];

        float y = 0.0f;
        for (int i = 0; i < N; i++) {
            const float x = X[x_index];

            y += x;

            x_index += x_target_axis_stride;
        }

        Y[gid] = y;
    }
}


void fusedelementwise_e580eb60fff7d9c089fcc408082bcbc2ad574fe552e494230c0426c6(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    float * v3 = (static_buffer + meta_buffer[2]);
    const int v4 = meta_buffer[3];
    const int v5 = meta_buffer[4];
    const int D0 = meta_buffer[5];
    const int D1 = meta_buffer[6];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        const float v6 = v2[d0];
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            float v7;
            {
                float value = *((float *)(&meta_buffer[7]));
                v7 = v6 * value;
            }
            float v8;
            {
                float value = *((float *)(&meta_buffer[8]));
                v8 = v7 + value;
            }
            float v9;
            {
                float value = *((float *)(&meta_buffer[9]));
                v9 = powf(v8, value);
            }
            const float v10 = v1[d0 + d1*v4];
            float v11;
            {
                v11 = v10 / v9;
            }
            float v12;
            {
                float slope = *((float *)(&meta_buffer[10]));
                v12 = v11 > 0 ? v11 : (v11 * slope);
            }
            v3[d0*v5 + d1] = v12;
        }
    }
}


void unpooling2d_7ab39179ab820eb2476f7a86b7963ff313ca52336b01bc45cbe4a952(const int * meta_buffer)
{
    const float *X = (static_buffer + meta_buffer[0]);
    float *Y = (static_buffer + meta_buffer[1]);
    const int N = meta_buffer[2];
    const int H1 = meta_buffer[3];
    const int W1 = meta_buffer[4];
    const int C = meta_buffer[5];
    const int H2 = meta_buffer[6];
    const int W2 = meta_buffer[7];

    const int KH = meta_buffer[8];
    const int KW = meta_buffer[9];
    const int SH = meta_buffer[10];
    const int SW = meta_buffer[11];
    const int PH = meta_buffer[12];
    const int PW = meta_buffer[13];

    for (int gid = 0; gid < N * H2 * W2 * C; gid += 1) {
        const int c = gid % C;
        const int w2 = gid / C % W2;
        const int h2 = gid / C / W2 % H2;
        const int n = gid / C / W2 / H2;

        float v = 0;
        for (int kh = 0; kh < KH; kh++) {
            int h1 = h2 + PH - kh;
            if (h1 < 0 || h1 >= H1 * SH) continue;
            if (h1 % SH != 0) continue;
            h1 /= SH;
            for (int kw = 0; kw < KW; kw++) {
                int w1 = w2 + PW - kw;
                if (w1 < 0 || w1 >= W1 * SW) continue;
                if (w1 % SW != 0) continue;
                w1 /= SW;
                v += X[((n * H1 + h1) * W1 + w1) * C + c];
            }
        }

        Y[gid] = v;
    }
}


void fusedelementwise_bf2429aa06e03fb0cf3d7c16d8beb4a76a2eeaf9b7283978776beaf7(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    float * v3 = (static_buffer + meta_buffer[2]);
    const int v4 = meta_buffer[3];
    const int v5 = meta_buffer[4];
    const int D0 = meta_buffer[5];
    const int D1 = meta_buffer[6];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        const float v6 = v2[d0];
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            float v7;
            {
                float value = *((float *)(&meta_buffer[7]));
                v7 = v6 * value;
            }
            float v8;
            {
                float value = *((float *)(&meta_buffer[8]));
                v8 = v7 + value;
            }
            float v9;
            {
                float value = *((float *)(&meta_buffer[9]));
                v9 = powf(v8, value);
            }
            const float v10 = v1[d0 + d1*v4];
            float v11;
            {
                v11 = v10 / v9;
            }
            float v12;
            {
                float slope = *((float *)(&meta_buffer[10]));
                v12 = v11 > 0 ? v11 : (v11 * slope);
            }
            float v13;
            {
                float value = *((float *)(&meta_buffer[11]));
                v13 = v12 * value;
            }
            v3[d0*v5 + d1] = v13;
        }
    }
}


void fusedelementwise_a539fc433f70acc39bfd392d237ee1ddac97fd280c5c3d0b7624788c(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    float * v3 = (static_buffer + meta_buffer[2]);
    const int v4 = meta_buffer[3];
    const int v5 = meta_buffer[4];
    const int D0 = meta_buffer[5];
    const int D1 = meta_buffer[6];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        const float v6 = v1[d0];
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            float v7;
            {
                float value = *((float *)(&meta_buffer[7]));
                v7 = v6 * value;
            }
            float v8;
            {
                float value = *((float *)(&meta_buffer[8]));
                v8 = v7 + value;
            }
            float v9;
            {
                float value = *((float *)(&meta_buffer[9]));
                v9 = powf(v8, value);
            }
            const float v10 = v2[d0 + d1*v4];
            float v11;
            {
                v11 = v10 / v9;
            }
            float v12;
            {
                float slope = *((float *)(&meta_buffer[10]));
                v12 = v11 > 0 ? v11 : (v11 * slope);
            }
            v3[d0*v5 + d1] = v12;
        }
    }
}


void fusedelementwise_577b0886dad6a4e0a445f105e7b36ed43f8f1f2b54b07c180cf81ab5(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    float * v3 = (static_buffer + meta_buffer[2]);
    const int v4 = meta_buffer[3];
    const int v5 = meta_buffer[4];
    const int D0 = meta_buffer[5];
    const int D1 = meta_buffer[6];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        const float v6 = v1[d0];
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            float v7;
            {
                float value = *((float *)(&meta_buffer[7]));
                v7 = v6 * value;
            }
            float v8;
            {
                float value = *((float *)(&meta_buffer[8]));
                v8 = v7 + value;
            }
            float v9;
            {
                float value = *((float *)(&meta_buffer[9]));
                v9 = powf(v8, value);
            }
            const float v10 = v2[d0 + d1*v4];
            float v11;
            {
                v11 = v10 / v9;
            }
            float v12;
            {
                float slope = *((float *)(&meta_buffer[10]));
                v12 = v11 > 0 ? v11 : (v11 * slope);
            }
            float v13;
            {
                float value = *((float *)(&meta_buffer[11]));
                v13 = v12 * value;
            }
            v3[d0*v5 + d1] = v13;
        }
    }
}


void fusedelementwise_1b5e9cb2fc3f81a00f9eb9c72ff0f34b4cf49d7018fd728b50be37a5(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    const float * v3 = (static_buffer + meta_buffer[2]);
    float * v4 = (static_buffer + meta_buffer[3]);
    const int v5 = meta_buffer[4];
    const int v6 = meta_buffer[5];
    const int v7 = meta_buffer[6];
    const int D0 = meta_buffer[7];
    const int D1 = meta_buffer[8];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        const float v8 = v1[d0];
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            const float v9 = v3[d0 + d1*v6];
            float v10;
            {
                v10 = v9 + v8;
            }
            const float v11 = v2[d0*v5 + d1];
            float v12;
            {
                v12 = v10 > v11 ? 1.0 : 0.0;
            }
            float v13;
            {
                float value = *((float *)(&meta_buffer[9]));
                v13 = v12 * value;
            }
            float v14;
            {
                const int value = meta_buffer[10];
                v14 = v13 + value;
            }
            float v15;
            {
                v15 = v10 * v14;
            }
            float v16;
            {
                v16 = v15 + v12;
            }
            v4[d0 + d1*v7] = v16;
        }
    }
}


void transpose_3ede431373fd6b14a21ca5a7f2fab289eb4c42220e346f22acc816e8(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    float * v2 = (static_buffer + meta_buffer[1]);
    const int v3 = meta_buffer[2];
    const int v4 = meta_buffer[3];
    const int D0 = meta_buffer[4];
    const int D1 = meta_buffer[5];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            const float v5 = v1[d0*v3 + d1];
            float v6;
            {
                v6 = v5;
            }
            v2[d0 + d1*v4] = v6;
        }
    }
}


void scalarmul_7c2c33c26c0830da12244b0c29c2f0ee677dc59705c4e472ec8bef37(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    float * v2 = (static_buffer + meta_buffer[1]);
    const int v3 = meta_buffer[2];
    const int v4 = meta_buffer[3];
    const int D0 = meta_buffer[4];
    const int D1 = meta_buffer[5];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            const float v5 = v1[d0*v3 + d1];
            float v6;
            {
                float value = *((float *)(&meta_buffer[6]));
                v6 = v5 * value;
            }
            v2[d0 + d1*v4] = v6;
        }
    }
}


void fusedelementwise_9d72c498e1f9b5796a7eb7dee52a406b1f6ac2c63c8cfb553fd901b6(const int * meta_buffer)
{
    const float * v1 = (static_buffer + meta_buffer[0]);
    const float * v2 = (static_buffer + meta_buffer[1]);
    const float * v3 = (static_buffer + meta_buffer[2]);
    const float * v4 = (static_buffer + meta_buffer[3]);
    const float * v5 = (static_buffer + meta_buffer[4]);
    float * v6 = (static_buffer + meta_buffer[5]);
    const int v7 = meta_buffer[6];
    const int v8 = meta_buffer[7];
    const int v9 = meta_buffer[8];
    const int v10 = meta_buffer[9];
    const int v11 = meta_buffer[10];
    const int D0 = meta_buffer[11];
    const int D1 = meta_buffer[12];
    int d0;
    for (d0 = ((1 > 8) ? (0 % (1 / 8)) : 0); d0 < D0; d0 += ((1 > 8) ? (1 / 8) : 1)) {
        const float v12 = v1[d0];
        int d1;
        for (d1 = ((1 > 8) ? (0 / (1 / 8)) : 0); d1 < D1; d1 += ((1 > 8) ? 8 : 1)) {
            const float v13 = v4[d0*v9 + d1];
            const float v14 = v5[d0 + d1*v10];
            float v15;
            {
                v15 = v14 + v13;
            }
            float v16;
            {
                v16 = v12 + v15;
            }
            const float v17 = v3[d0*v8 + d1];
            float v18;
            {
                v18 = v16 > v17 ? 1.0 : 0.0;
            }
            float v19;
            {
                float value = *((float *)(&meta_buffer[13]));
                v19 = v18 * value;
            }
            float v20;
            {
                const int value = meta_buffer[14];
                v20 = v19 + value;
            }
            float v21;
            {
                v21 = v16 * v20;
            }
            float v22;
            {
                v22 = v21 + v18;
            }
            const float v23 = v2[d0*v7 + d1];
            float v24;
            {
                v24 = v22 > v23 ? 1.0 : 0.0;
            }
            float v25;
            {
                v25 = v22 * v24;
            }
            float v26;
            {
                float value = *((float *)(&meta_buffer[15]));
                v26 = v24 * value;
            }
            float v27;
            {
                const int value = meta_buffer[16];
                v27 = v26 + value;
            }
            float v28;
            {
                v28 = v23 * v27;
            }
            float v29;
            {
                v29 = v25 + v28;
            }
            v6[d0 + d1*v11] = v29;
        }
    }
}


void averagepooling2d_315133a9f01ff68a853da10590f9591258950d2d5ab8b7bac49ab0ea(const int * meta_buffer)
{
    const float *X = (static_buffer + meta_buffer[0]);
    float *Y = (static_buffer + meta_buffer[1]);
    const int N = meta_buffer[2];
    const int H1 = meta_buffer[3];
    const int W1 = meta_buffer[4];
    const int C = meta_buffer[5];
    const int H2 = meta_buffer[6];
    const int W2 = meta_buffer[7];

    const int KH = meta_buffer[8];
    const int KW = meta_buffer[9];
    const int SH = meta_buffer[10];
    const int SW = meta_buffer[11];
    const int PH = meta_buffer[12];
    const int PW = meta_buffer[13];

    for (int gid = 0; gid < N * H2 * W2 * C; gid += 1) {
        const int c = gid % C;
        const int w2 = gid / C % W2;
        const int h2 = gid / C / W2 % H2;
        const int n = gid / C / W2 / H2;

        float v = 0;
        
        for (int kh = 0; kh < KH; kh++) {
            const int h1 = h2 * SH - PH + kh;
            if (h1 < 0 || h1 >= H1) continue;

            for (int kw = 0; kw < KW; kw++) {
                const int w1 = w2 * SW - PW + kw;
                if (w1 < 0 || w1 >= W1) continue;

                v += X[((n * H1 + h1) * W1 + w1) * C + c];
                
            }
        }
        v /= KH * KW;

        Y[gid] = v;
    }
}

extern "C" void run() {
reshape_f6c88dc1fd9479f62d789912530e6a5c7c51c73de368540e8b8e1ceb(meta_buf_0);
scalarmul_021226a8fb40f401fcd74ecc98cab18a4513f970372147bec0077349(meta_buf_1);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_2);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_3);
fusedelementwise_99aebe006807faa2c23dcb393bb85a063a705d397c501de9b47249da(meta_buf_4);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_5);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_6);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_7);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_8);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_9);
fusedelementwise_e580eb60fff7d9c089fcc408082bcbc2ad574fe552e494230c0426c6(meta_buf_10);
unpooling2d_7ab39179ab820eb2476f7a86b7963ff313ca52336b01bc45cbe4a952(meta_buf_11);
scalarmul_021226a8fb40f401fcd74ecc98cab18a4513f970372147bec0077349(meta_buf_12);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_13);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_14);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_15);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_16);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_17);
fusedelementwise_bf2429aa06e03fb0cf3d7c16d8beb4a76a2eeaf9b7283978776beaf7(meta_buf_18);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_19);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_20);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_21);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_22);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_23);
fusedelementwise_a539fc433f70acc39bfd392d237ee1ddac97fd280c5c3d0b7624788c(meta_buf_24);
unpooling2d_7ab39179ab820eb2476f7a86b7963ff313ca52336b01bc45cbe4a952(meta_buf_25);
scalarmul_021226a8fb40f401fcd74ecc98cab18a4513f970372147bec0077349(meta_buf_26);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_27);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_28);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_29);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_30);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_31);
fusedelementwise_577b0886dad6a4e0a445f105e7b36ed43f8f1f2b54b07c180cf81ab5(meta_buf_32);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_33);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_34);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_35);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_36);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_37);
fusedelementwise_a539fc433f70acc39bfd392d237ee1ddac97fd280c5c3d0b7624788c(meta_buf_38);
unpooling2d_7ab39179ab820eb2476f7a86b7963ff313ca52336b01bc45cbe4a952(meta_buf_39);
scalarmul_021226a8fb40f401fcd74ecc98cab18a4513f970372147bec0077349(meta_buf_40);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_41);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_42);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_43);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_44);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_45);
fusedelementwise_577b0886dad6a4e0a445f105e7b36ed43f8f1f2b54b07c180cf81ab5(meta_buf_46);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_47);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_48);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_49);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_50);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_51);
fusedelementwise_a539fc433f70acc39bfd392d237ee1ddac97fd280c5c3d0b7624788c(meta_buf_52);
unpooling2d_7ab39179ab820eb2476f7a86b7963ff313ca52336b01bc45cbe4a952(meta_buf_53);
scalarmul_021226a8fb40f401fcd74ecc98cab18a4513f970372147bec0077349(meta_buf_54);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_55);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_56);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_57);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_58);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_59);
fusedelementwise_577b0886dad6a4e0a445f105e7b36ed43f8f1f2b54b07c180cf81ab5(meta_buf_60);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_61);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_62);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_63);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_64);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_65);
fusedelementwise_a539fc433f70acc39bfd392d237ee1ddac97fd280c5c3d0b7624788c(meta_buf_66);
unpooling2d_7ab39179ab820eb2476f7a86b7963ff313ca52336b01bc45cbe4a952(meta_buf_67);
scalarmul_021226a8fb40f401fcd74ecc98cab18a4513f970372147bec0077349(meta_buf_68);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_69);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_70);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_71);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_72);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_73);
fusedelementwise_577b0886dad6a4e0a445f105e7b36ed43f8f1f2b54b07c180cf81ab5(meta_buf_74);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_75);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_76);
elementwiseadd_a7a1bac0d1004f08225f89ef90716997e916fb65d62d759a6fac66b7(meta_buf_77);
scalarpow_4163f9148e20f2a5cb3b6e2a7fccc465b35d9b347dc8f77b9766689a(meta_buf_78);
sum_c54590818bcfdc4feb25405117073fe986b83ec6709a9e873cedff99(meta_buf_79);
fusedelementwise_577b0886dad6a4e0a445f105e7b36ed43f8f1f2b54b07c180cf81ab5(meta_buf_80);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_81);
fusedelementwise_1b5e9cb2fc3f81a00f9eb9c72ff0f34b4cf49d7018fd728b50be37a5(meta_buf_82);
unpooling2d_7ab39179ab820eb2476f7a86b7963ff313ca52336b01bc45cbe4a952(meta_buf_83);
transpose_3ede431373fd6b14a21ca5a7f2fab289eb4c42220e346f22acc816e8(meta_buf_84);
scalarmul_7c2c33c26c0830da12244b0c29c2f0ee677dc59705c4e472ec8bef37(meta_buf_85);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_86);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_87);
fusedelementwise_99aebe006807faa2c23dcb393bb85a063a705d397c501de9b47249da(meta_buf_88);
im2col_54f86552263d1a348fd988a8122143d1a6ec1c1aa7867ba5833442fb(meta_buf_89);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_90);
fusedelementwise_99aebe006807faa2c23dcb393bb85a063a705d397c501de9b47249da(meta_buf_91);
tensordot_b0c87b9ad9c24d5c638538a68fd9e1ad2b472c70c0e98883fad5c6af(meta_buf_92);
fusedelementwise_9d72c498e1f9b5796a7eb7dee52a406b1f6ac2c63c8cfb553fd901b6(meta_buf_93);
averagepooling2d_315133a9f01ff68a853da10590f9591258950d2d5ab8b7bac49ab0ea(meta_buf_94);
transpose_3ede431373fd6b14a21ca5a7f2fab289eb4c42220e346f22acc816e8(meta_buf_95);

}

