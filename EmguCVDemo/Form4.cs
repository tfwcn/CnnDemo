using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using EmguCVDemo.BP;

namespace EmguCVDemo
{
    public partial class Form4 : Form
    {
        //private int bpWidth = 960, bpHeight = 540;
        private int bpWidth = 1920, bpHeight = 1080;
        private int bpRectangleCount = 50;//人脸框最大数量
        private int bpTrainDataCount;//训练样本数
        private Ann ann = new Ann();
        private Cnn cnn = new Cnn();
        public Form4()
        {
            InitializeComponent();
        }

        private void Form4_Load(object sender, EventArgs e)
        {
            /*int[] layerSizes = new int[] { 
                bpWidth * bpHeight,
                bpWidth * bpHeight+100, 50, 30, 10,
                //20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                //20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                bpRectangleCount * 4 
            };
            int[] layerType = new int[] { 
                1,
                1, 1, 1, 1,
                //20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                //20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                1
            };
            ann.SetLayerSizes(layerSizes,
                layerType);*/
            cnn.AddCnnConvolutionLayer(8, bpWidth, bpHeight, 10, 10, 2, 2, 1, 2, 2, 1);
            cnn.AddCnnConvolutionLayer(20, 5, 5, 1, 1, 1, 2, 2, 1);
            cnn.AddCnnFullLayer(300, 1);
            cnn.AddCnnFullLayer(50 * 4, 1);
        }
    }
}
