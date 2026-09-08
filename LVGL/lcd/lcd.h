#include "c8051F340.h"
#include <main.h>

#ifndef __LCD_h__
#define __LCD_h__

#ifdef DEBUG
#define END 0xffff
#endif

extern unsigned int code debug_code[50][15];
extern unsigned char  code  ascii[]; 
extern unsigned int   code  pic_eval[];
//---------------------------------------------------------------------
void CS_CLR(void);
void CS_SET(void);
void SCK_CLR(void);
void SCK_SET(void);
void SDI_CLR(void);
void SDI_SET(void);
void RST_CLR(void);
void RST_SET(void);
void SPI_SendData(unsigned char i);
void SPI_WriteComm(unsigned char i);
void SPI_WriteData(unsigned char i);
void LCD_Init(void);

void WriteComm(unsigned int i);
void WriteData(unsigned int i);
void WriteDispData(unsigned char dataH,unsigned char dataL);

void WriteDispData(unsigned char DataH,unsigned char DataL);
void BlockWrite(unsigned int Xstart,unsigned int Xend,unsigned int Ystart,unsigned int Yend) reentrant;
void DispColor(unsigned int color);
void DispBand(void);
void DispFrame(void);
void DispPic(unsigned int code *picture);
void DispPicFromSD(unsigned char PicNum);

void DispScaleHor1(void);
void DispScaleVer(void);
void DispScaleVer_Red(void);
void DispScaleVer_Green(void);
void DispScaleVer_Blue(void);
void DispScaleVer_Gray(void);
void DispScaleHor2(void);
void DispSnow(void);
void DispBlock(void);
void displaygray1(unsigned int gray);

void WriteOneDot(unsigned int color);
unsigned char ToOrd(unsigned char ch); 
void DispOneChar(unsigned char ord,unsigned int Xstart,unsigned int Ystart,unsigned int TextColor,unsigned int BackColor);
void DispStr(unsigned char *str,unsigned int Xstart,unsigned int Ystart,unsigned int TextColor,unsigned int BackColor);
void DispInt(unsigned int i,unsigned int Xstart,unsigned int Ystart,unsigned int TextColor,unsigned int BackColor);

unsigned int ReadData(void);
void DispRegValue(unsigned int RegIndex,unsigned char ParNum);

void Debug(void);

void PutPixel(unsigned int x,unsigned int y,unsigned int color);
void DrawLine(unsigned int Xstart,unsigned int Xend,unsigned int Ystart,unsigned int Yend,unsigned int color);
void DrawGird(unsigned int color);
void GC9106_VScroll_Mode(unsigned int i);
#endif
