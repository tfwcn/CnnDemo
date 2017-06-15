using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using EmguCVDemo.BP;
using Emgu.CV;
using Emgu.CV.Structure;

namespace EmguCVDemo
{
    public partial class Form4 : Form
    {
        private Cnn cnn;
        public Form4()
        {
            InitializeComponent();
        }

        private void Form4_Load(object sender, EventArgs e)
        {
            cnn = new Cnn();
            cnn.AddCnnConvolutionLayer(20, 28, 28, 5, 5, 1, 1, 1, 2, 2, 1);
            cnn.AddCnnConvolutionLayer(20, 5, 5, 1, 1, 1, 2, 2, 1);
            cnn.AddCnnFullLayer(100, 1);
            cnn.AddCnnFullLayer(10, 1);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            FileStream fs = new FileStream("train-labels.idx1-ubyte", FileMode.Open);
            FileStream fsImages = new FileStream("train-images.idx3-ubyte", FileMode.Open);
            byte[] bytes4 = new byte[4];
            fsImages.Seek(4, SeekOrigin.Current);
            fs.Seek(8, SeekOrigin.Current);
            fsImages.Read(bytes4, 0, 4);
            int count = ToInt32(bytes4);
            fsImages.Read(bytes4, 0, 4);
            int height = ToInt32(bytes4);
            fsImages.Read(bytes4, 0, 4);
            int width = ToInt32(bytes4);
            //for (int i = 0; i < count; i++)
            for (int i = 0; i < 10000; i++)
            {
                Bitmap img = GetImage(fsImages, width, height);
                byte label = GetLable(fs);
                double[] labels = new double[10];
                for (int i2 = 0; i2 < 10; i2++)
                {
                    labels[i2] = -1;
                }
                labels[label] = 1;
                Image<Bgr, float> trainingData = new Image<Bgr, float>(img); ;
                double[,] input = new double[width, height];
                for (int w = 0; w < width; w++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        input[w, h] = Color.FromArgb(
                            (int)trainingData.Data[h, w, 2],
                            (int)trainingData.Data[h, w, 1],
                            (int)trainingData.Data[h, w, 0]
                            ).ToArgb() / (float)0xFFFFFF;
                    }
                }
                cnn.Train(input, labels, 0.01);
                //img.Save("imgs/" + i + "_" + label + ".jpg");
            }
            fsImages.Close();
            fs.Close();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            FileStream fs = new FileStream("t10k-labels.idx1-ubyte", FileMode.Open);
            FileStream fsImages = new FileStream("t10k-images.idx3-ubyte", FileMode.Open);
            byte[] bytes4 = new byte[4];
            fsImages.Seek(4, SeekOrigin.Current);
            fs.Seek(8, SeekOrigin.Current);
            fsImages.Read(bytes4, 0, 4);
            int count = ToInt32(bytes4);
            fsImages.Read(bytes4, 0, 4);
            int height = ToInt32(bytes4);
            fsImages.Read(bytes4, 0, 4);
            int width = ToInt32(bytes4);
            for (int i = 0; i < count; i++)
            {
                Bitmap img = GetImage(fsImages, width, height);
                byte label = GetLable(fs);
                Image<Bgr, float> trainingData = new Image<Bgr, float>(img); ;
                double[,] input = new double[width, height];
                for (int w = 0; w < width; w++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        input[w, h] = Color.FromArgb(
                            (int)trainingData.Data[h, w, 2],
                            (int)trainingData.Data[h, w, 1],
                            (int)trainingData.Data[h, w, 0]
                            ).ToArgb() / (float)0xFFFFFF;
                    }
                }
                double[] labels = cnn.Predict(input);
                double maxtype = labels[0], max = 0;
                for (int n = 0; n < 10; n++)
                {
                    if (maxtype < labels[n])
                    {
                        max = n;
                        maxtype = labels[n];
                    }
                }
                Console.WriteLine(i + ":" + label + "," + max);
                //img.Save("imgs/" + i + "_" + label + ".jpg");
            }
            fsImages.Close();
            fs.Close();
        }

        private Bitmap GetImage(FileStream fs, int width, int height)
        {
            Bitmap img = new Bitmap(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte[] bytes1 = new byte[1];
                    fs.Read(bytes1, 0, 1);
                    img.SetPixel(x, y, Color.FromArgb(255 - bytes1[0], 255 - bytes1[0], 255 - bytes1[0]));
                }
            }
            return img;
        }

        private byte GetLable(FileStream fs)
        {
            byte[] bytes1 = new byte[1];
            fs.Read(bytes1, 0, 1);
            return bytes1[0];
        }

        private int ToInt32(byte[] bytes)
        {
            int count = bytes[3];
            count += bytes[2] << 8;
            count += bytes[1] << 16;
            count += bytes[0] << 24;
            return count;
        }
    }
}
