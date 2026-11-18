#include <cstdint>
#include <cstring>
#include <cmath>
static float half(const unsigned char* p){_Float16 v;std::memcpy(&v,p,2);return float(v);}
extern "C" int decode(int type,const unsigned char* src,uint64_t count,uint16_t* out){
 const int block=type==0||type==1?1:256,bytes=type==0?4:type==1?2:type==12?144:type==14?210:0;
 if(!bytes||count%block)return 1;
 for(uint64_t b=0;b<count/block;b++,src+=bytes){
  float values[256];
  if(type==0)std::memcpy(values,src,4);
  else if(type==1)values[0]=half(src);
  else if(type==12){
   float d=half(src),m=half(src+2);const auto*s=src+4;const auto*q=src+16;
   for(int group=0;group<8;group++){
    int sc=group<4?s[group]&63:(s[group+4]&15)|((s[group-4]>>6)<<4);
    int mn=group<4?s[group+4]&63:(s[group+4]>>4)|((s[group]>>6)<<4);
    for(int l=0;l<32;l++){int v=(q[(group/2)*32+l]>>(group%2*4))&15;values[group*32+l]=(d*sc)*v-m*mn;}
   }
  }else{
   float d=half(src+208);auto*sc=reinterpret_cast<const int8_t*>(src+192);
   for(int i=0;i<256;i++){
    int n=i/128,l=i%32,section=(i%128)/32;
    int low=(src[n*64+l+(section%2)*32]>>(section/2*4))&15;
    int high=(src[128+n*32+l]>>(section*2))&3;
    values[i]=(d*sc[i/16])*((low|(high<<4))-32);
   }
  }
  for(int i=0;i<block;i++){
   _Float16 h=static_cast<_Float16>(values[i]);if(!std::isfinite(values[i])||!std::isfinite(float(h)))return 2;
   std::memcpy(out+b*block+i,&h,2);
  }
 }
 return 0;
}
