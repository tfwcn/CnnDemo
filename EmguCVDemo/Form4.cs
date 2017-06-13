using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;

namespace EmguCVDemo
{
    public partial class Form4 : Form
    {
        public Form4()
        {
            InitializeComponent();
        }

        private void Form4_Load(object sender, EventArgs e)
        {
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
            for (int i = 0; i < count; i++)
            {
                Bitmap img = GetImage(fsImages, width, height);
                byte label = GetLable(fs);
                img.Save("imgs/" + i + "_" + label + ".jpg");
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
