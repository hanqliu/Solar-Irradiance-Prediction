{\rtf1\ansi\ansicpg936\cocoartf2638
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Monaco;\f1\fnil\fcharset134 PingFangSC-Regular;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red255\green255\blue255;\red135\green135\blue135;
\red193\green193\blue193;\red245\green186\blue68;\red191\green131\blue194;\red109\green188\blue135;\red234\green121\blue57;
\red88\green196\blue193;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\csgray\c100000;\cssrgb\c60000\c60000\c60000;
\cssrgb\c80000\c80000\c80000;\cssrgb\c97255\c77255\c33333;\cssrgb\c80000\c60000\c80392;\cssrgb\c49412\c77647\c60000;\cssrgb\c94118\c55294\c28627;
\cssrgb\c40392\c80392\c80000;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec4 // 
\f1 \'d2\'bb\'ce\'ac
\f0 FFT\strokec5 \
\pard\pardeftab720\partightenfactor0
\cf2 \strokec6 #\strokec7 include\strokec6  \strokec8 "cuda_runtime.h"\strokec5 \
\strokec6 #\strokec7 include\strokec6  \strokec8 "device_launch_parameters.h"\strokec5 \
\strokec6 #\strokec7 include\strokec6  \strokec8 "complex.cu"\strokec5 \
\strokec6 #\strokec7 include\strokec6  \strokec8 <iostream>\strokec5 \
\strokec6 #\strokec7 include\strokec6  \strokec8 <string>\strokec5 \
\strokec6 #\strokec7 include\strokec6  \strokec8 <stdlib.h>\strokec5 \
\strokec6 #\strokec7 include\strokec6  \strokec8 <time.h>\strokec5 \
\strokec6 #\strokec7 include\strokec6  \strokec8 <Windows.h>\strokec5 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec7 int\strokec5  \strokec9 GetBits\strokec5 (\strokec7 int\strokec5  n) \{\
    \strokec7 int\strokec5  bits \strokec10 =\strokec5  \strokec9 0\strokec5 ;\
    \strokec7 while\strokec5  (n \strokec10 >>=\strokec5  \strokec9 1\strokec5 ) \{\
        bits\strokec10 ++\strokec5 ;\
    \}\
    \strokec7 return\strokec5  bits;\
\}\
\
__device__ \strokec7 int\strokec5  \strokec9 BinaryReverse\strokec5 (\strokec7 int\strokec5  i, \strokec7 int\strokec5  bits) \{\
    \strokec7 int\strokec5  r \strokec10 =\strokec5  \strokec9 0\strokec5 ;\
    \strokec7 do\strokec5  \{\
        r \strokec10 +=\strokec5  i \strokec10 %\strokec5  \strokec9 2\strokec5  \strokec10 <<\strokec5  \strokec10 --\strokec5 bits;\
    \} \strokec7 while\strokec5  (i \strokec10 /=\strokec5  \strokec9 2\strokec5 );\
    \strokec7 return\strokec5  r;\
\}\
\
__device__ \strokec7 void\strokec5  \strokec9 Bufferfly\strokec5 (Complex \strokec10 *\strokec5 a, Complex \strokec10 *\strokec5 b, Complex factor) \{\
    Complex a1 \strokec10 =\strokec5  (\strokec10 *\strokec5 a) \strokec10 +\strokec5  factor \strokec10 *\strokec5  (\strokec10 *\strokec5 b);\
    Complex b1 \strokec10 =\strokec5  (\strokec10 *\strokec5 a) \strokec10 -\strokec5  factor \strokec10 *\strokec5  (\strokec10 *\strokec5 b);\
    \strokec10 *\strokec5 a \strokec10 =\strokec5  a1;\
    \strokec10 *\strokec5 b \strokec10 =\strokec5  b1;\
\}\
\
__global__ \strokec7 void\strokec5  \strokec9 FFT\strokec5 (Complex nums[], Complex result[], \strokec7 int\strokec5  n, \strokec7 int\strokec5  bits) \{\
    \strokec7 int\strokec5  tid \strokec10 =\strokec5  threadIdx.x \strokec10 +\strokec5  blockDim.x \strokec10 *\strokec5  blockIdx.x;\
    \strokec7 if\strokec5  (tid \strokec10 >=\strokec5  n) \strokec7 return\strokec5 ;\
    \strokec7 for\strokec5  (\strokec7 int\strokec5  i \strokec10 =\strokec5  \strokec9 2\strokec5 ; i \strokec10 <\strokec5  \strokec9 2\strokec5  \strokec10 *\strokec5  n; i \strokec10 *=\strokec5  \strokec9 2\strokec5 ) \{\
        \strokec7 if\strokec5  (tid \strokec10 %\strokec5  i \strokec10 ==\strokec5  \strokec9 0\strokec5 ) \{\
            \strokec7 int\strokec5  k \strokec10 =\strokec5  i;\
            \strokec7 if\strokec5  (n \strokec10 -\strokec5  tid \strokec10 <\strokec5  k) k \strokec10 =\strokec5  n \strokec10 -\strokec5  tid;\
            \strokec7 for\strokec5  (\strokec7 int\strokec5  j \strokec10 =\strokec5  \strokec9 0\strokec5 ; j \strokec10 <\strokec5  k \strokec10 /\strokec5  \strokec9 2\strokec5 ; \strokec10 ++\strokec5 j) \{\
                \strokec9 Bufferfly\strokec5 (\strokec10 &\strokec5 nums[\strokec9 BinaryReverse\strokec5 (tid \strokec10 +\strokec5  j, bits)], \strokec10 &\strokec5 nums[\strokec9 BinaryReverse\strokec5 (tid \strokec10 +\strokec5  j \strokec10 +\strokec5  k \strokec10 /\strokec5  \strokec9 2\strokec5 , bits)], Complex\strokec10 ::\strokec9 W\strokec5 (k, j));\
            \}\
        \}\
        \strokec9 __syncthreads\strokec5 ();\
    \}\
    result[tid] \strokec10 =\strokec5  nums[\strokec9 BinaryReverse\strokec5 (tid, bits)];\
\}\
\
\strokec7 void\strokec5  \strokec9 printSequence\strokec5 (Complex nums[], \strokec7 const\strokec5  \strokec7 int\strokec5  N) \{\
    \strokec9 printf\strokec5 (\strokec8 "["\strokec5 );\
    \strokec7 for\strokec5  (\strokec7 int\strokec5  i \strokec10 =\strokec5  \strokec9 0\strokec5 ; i \strokec10 <\strokec5  N; \strokec10 ++\strokec5 i) \{\
        \strokec7 double\strokec5  real \strokec10 =\strokec5  nums[i].real, imag \strokec10 =\strokec5  nums[i].imag;\
        \strokec7 if\strokec5  (imag \strokec10 ==\strokec5  \strokec9 0\strokec5 ) \strokec9 printf\strokec5 (\strokec8 "%.16f"\strokec5 , real);\
        \strokec7 else\strokec5  \{\
            \strokec7 if\strokec5  (imag \strokec10 >\strokec5  \strokec9 0\strokec5 ) \strokec9 printf\strokec5 (\strokec8 "%.16f+%.16fi"\strokec5 , real, imag);\
            \strokec7 else\strokec5  \strokec9 printf\strokec5 (\strokec8 "%.16f%.16fi"\strokec5 , real, imag);\
        \}\
        \strokec7 if\strokec5  (i \strokec10 !=\strokec5  N \strokec10 -\strokec5  \strokec9 1\strokec5 ) \strokec9 printf\strokec5 (\strokec8 ", "\strokec5 );\
    \}\
    \strokec9 printf\strokec5 (\strokec8 "]\\n"\strokec5 );\
\}\
\
\strokec7 int\strokec5  \strokec9 main\strokec5 () \{\
    \strokec9 srand\strokec5 (\strokec9 time\strokec5 (\strokec9 0\strokec5 ));\
    \strokec7 const\strokec5  \strokec7 int\strokec5  TPB \strokec10 =\strokec5  \strokec9 1024\strokec5 ;\
    \strokec7 const\strokec5  \strokec7 int\strokec5  N \strokec10 =\strokec5  \strokec9 1024\strokec5  \strokec10 *\strokec5  \strokec9 32\strokec5 ; \
    \strokec7 const\strokec5  \strokec7 int\strokec5  bits \strokec10 =\strokec5  \strokec9 GetBits\strokec5 (N);\
    \
    Complex \strokec10 *\strokec5 nums \strokec10 =\strokec5  (Complex\strokec10 *\strokec5 )\strokec9 malloc\strokec5 (\strokec7 sizeof\strokec5 (Complex) \strokec10 *\strokec5  N), \strokec10 *\strokec5 dNums, \strokec10 *\strokec5 dResult;\
    \strokec7 for\strokec5  (\strokec7 int\strokec5  i \strokec10 =\strokec5  \strokec9 0\strokec5 ; i \strokec10 <\strokec5  N; \strokec10 ++\strokec5 i) \{\
        nums[i] \strokec10 =\strokec5  Complex\strokec10 ::\strokec9 GetRandomReal\strokec5 ();\
    \}\
    \strokec9 printf\strokec5 (\strokec8 "Length of Sequence: %d\\n"\strokec5 , N);\
    \strokec4 printf("Before FFT: \\n");\strokec5 \
    \strokec4 printSequence(nums, N);\strokec5 \
    \
    \strokec7 float\strokec5  s \strokec10 =\strokec5  \strokec9 GetTickCount\strokec5 ();\
    \
    \strokec9 cudaMalloc\strokec5 ((\strokec7 void\strokec10 **\strokec5 )\strokec10 &\strokec5 dNums, \strokec7 sizeof\strokec5 (Complex) \strokec10 *\strokec5  N);\
    \strokec9 cudaMalloc\strokec5 ((\strokec7 void\strokec10 **\strokec5 )\strokec10 &\strokec5 dResult, \strokec7 sizeof\strokec5 (Complex) \strokec10 *\strokec5  N);\
    \strokec9 cudaMemcpy\strokec5 (dNums, nums, \strokec7 sizeof\strokec5 (Complex) \strokec10 *\strokec5  N, cudaMemcpyHostToDevice);\
    \
    dim3 threadPerBlock \strokec10 =\strokec5  \strokec9 dim3\strokec5 (TPB);\
    dim3 blockNum \strokec10 =\strokec5  \strokec9 dim3\strokec5 ((N \strokec10 +\strokec5  threadPerBlock.x \strokec10 -\strokec5  \strokec9 1\strokec5 ) \strokec10 /\strokec5  threadPerBlock.x);\
    FFT\strokec10 <<<\strokec5 blockNum, threadPerBlock\strokec10 >>>\strokec5 (dNums, dResult, N, bits);\
\
    \strokec9 cudaMemcpy\strokec5 (nums, dResult, \strokec7 sizeof\strokec5 (Complex) \strokec10 *\strokec5  N, cudaMemcpyDeviceToHost);\
    \
\
    \strokec7 float\strokec5  cost \strokec10 =\strokec5  \strokec9 GetTickCount\strokec5 () \strokec10 -\strokec5  s;\
    \strokec4 printf("After FFT: \\n");\strokec5 \
    \strokec4 printSequence(nums, N);\strokec5 \
    \strokec9 printf\strokec5 (\strokec8 "Time of Transfromation: %fms"\strokec5 , cost);\
 \
    \strokec9 free\strokec5 (nums);\
    \strokec9 cudaFree\strokec5 (dNums);\
    \strokec9 cudaFree\strokec5 (dResult);\
\}\
}