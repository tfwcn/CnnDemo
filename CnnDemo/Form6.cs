using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace CnnDemo
{
    public partial class Form6 : Form
    {
        public Form6()
        {
            InitializeComponent();
        }

        private void Form6_Load(object sender, EventArgs e)
        {
            try
            {
                //图1
                {
                    /*Bitmap img = new Bitmap(200, 200);
                    Graphics g = Graphics.FromImage(img);
                    g.Clear(Color.White);
                    g.DrawRectangle(Pens.Red, 10, 10, 1, 1);
                    g.DrawRectangle(Pens.Red, 100, 100, 1, 1);
                    g.DrawRectangle(Pens.Red, 10, 100, 1, 1);
                    g.DrawRectangle(Pens.Red, 100, 10, 1, 1);
                    g.Dispose();*/
                    Bitmap img = CnnDemo.CNN.CnnHelper.ImageZoom(new Bitmap("C:\\Users\\Administrator\\Desktop\\九楼人员旧照片收集\\曾昭奈.jpg"), 200, 200);
                    pbImg.Image = img;
                    //*
                    double[,] transformer = new double[3, 2];//3列2行
                    //旋转
                    transformer[0, 0] = Math.Cos(30 * 2 * Math.PI / 360);
                    transformer[1, 0] = Math.Sin(30 * 2 * Math.PI / 360);
                    transformer[2, 0] = 0;
                    transformer[0, 1] = -Math.Sin(30 * 2 * Math.PI / 360);
                    transformer[1, 1] = Math.Cos(30 * 2 * Math.PI / 360);
                    transformer[2, 1] = 0;
                    Bitmap img2 = CnnDemo.CNN.CnnHelper.ImageTransform(img, transformer);
                    pbImg2.Image = img2;
                    //*/
                }
                //图2
                {
                    double[,] transformer = new double[3, 2];//3列2行
                    //平移
                    //transformer[0, 0] = 1;
                    //transformer[1, 0] = 0;
                    //transformer[2, 0] = 50;//x平移
                    //transformer[0, 1] = 0;
                    //transformer[1, 1] = 1;
                    //transformer[2, 1] = 20;//y平移
                    //缩放
                    //transformer[0, 0] = 1.5;//x缩放
                    //transformer[1, 0] = 0;
                    //transformer[2, 0] = 0;
                    //transformer[0, 1] = 0;
                    //transformer[1, 1] = 1.5;//y缩放
                    //transformer[2, 1] = 0;
                    //旋转
                    //transformer[0, 0] = Math.Cos(30*2*Math.PI/360);
                    //transformer[1, 0] = Math.Sin(30*2*Math.PI/360);
                    //transformer[2, 0] = 0;
                    //transformer[0, 1] = -Math.Sin(30 * 2 * Math.PI / 360);
                    //transformer[1, 1] = Math.Cos(30*2*Math.PI/360);
                    //transformer[2, 1] = 0;

                    /*
                    double[,] point = new double[1, 3];//1列3行
                    Bitmap img = new Bitmap(200, 200);
                    Graphics g = Graphics.FromImage(img);
                    g.Clear(Color.White);
                    point[0, 0] = 10;
                    point[0, 1] = 10;
                    point[0, 2] = 1;
                    double[,] retPoint = CNN.CnnHelper.MatrixMultiply(transformer, point);
                    g.DrawRectangle(Pens.Red, (int)retPoint[0, 0], (int)retPoint[0, 1], 1, 1);
                    point[0, 0] = 100;
                    point[0, 1] = 100;
                    point[0, 2] = 1;
                    retPoint = CNN.CnnHelper.MatrixMultiply(transformer, point);
                    g.DrawRectangle(Pens.Red, (int)retPoint[0, 0], (int)retPoint[0, 1], 1, 1);
                    point[0, 0] = 10;
                    point[0, 1] = 100;
                    point[0, 2] = 1;
                    retPoint = CNN.CnnHelper.MatrixMultiply(transformer, point);
                    g.DrawRectangle(Pens.Red, (int)retPoint[0, 0], (int)retPoint[0, 1], 1, 1);
                    point[0, 0] = 100;
                    point[0, 1] = 10;
                    point[0, 2] = 1;
                    retPoint = CNN.CnnHelper.MatrixMultiply(transformer, point);
                    g.DrawRectangle(Pens.Red, (int)retPoint[0, 0], (int)retPoint[0, 1], 1, 1);
                    g.Dispose();
                    pbImg2.Image = img;
                    //*/
                }
            }
            catch (Exception ex)
            {

            }
        }
    }
}
