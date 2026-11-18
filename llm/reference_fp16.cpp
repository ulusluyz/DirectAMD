#include <cstdint>
#include <cstring>
extern "C" void reference_gemv(const uint16_t* weights,const double* x,double* out,uint64_t rows,uint64_t columns){
 static double lookup[65536];static bool ready=false;
 if(!ready){for(unsigned i=0;i<65536;i++){uint16_t bits=i;_Float16 h;std::memcpy(&h,&bits,2);lookup[i]=double(h);}ready=true;}
 for(uint64_t i=0;i<rows;i++){double sum=0;for(uint64_t j=0;j<columns;j++)sum+=lookup[weights[i*columns+j]]*x[j];out[i]=sum;}
}
