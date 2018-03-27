using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Emgu.CV.ML;
using Emgu.CV;
using System.Runtime.InteropServices;
using Emgu.CV.Structure;
using System.IO;

namespace CnnDemo
{
    public partial class Form2 : Form
    {
        ANN_MLP bp;
        public Form2()
        {
            InitializeComponent();
            //初始化
            bp = new ANN_MLP();
            Matrix<int> layerSizes = new Matrix<int>(new int[] { 2, 2, 2, 2, 1 });
            bp.SetLayerSizes(layerSizes);
            bp.SetActivationFunction(ANN_MLP.AnnMlpActivationFunction.Gaussian, 0, 0);
            bp.TermCriteria = new MCvTermCriteria(10, 1.0e-8);
            //bp.BackpropWeightScale = 0.1;
            //bp.BackpropMomentumScale = 0.1;
            bp.SetTrainMethod(ANN_MLP.AnnMlpTrainMethod.Backprop, 0, 0);
            //训练
            float[,] labels = new float[,] {
            { 0 }, { 1 }, { 0 }, { 1 }
            };
            Matrix<float> labelsMats = new Matrix<float>(labels);
            //Matrix<float> labelsMats = new Matrix<float>(count, 1);
            //Matrix<float> labelsMats1 = labelsMats.GetRows(0, count >> 1, 1);
            //labelsMats1.SetValue(1);
            //Matrix<float> labelsMats2 = labelsMats.GetRows(count >> 1, count, 1);
            //labelsMats2.SetValue(0);
            float[,] trainingData = new float[,] {
            { 1, 2 }, { 51, 52 }, { 111, 112 }, { 211, 212 }
            };
            for (int i = 0; i < trainingData.GetLength(0); i++)//归一化
            {
                for (int j = 0; j < trainingData.GetLength(1); j++)
                {
                    trainingData[i, j] /= 512;
                }
            }
            Matrix<float> trainingDataMat = new Matrix<float>(trainingData);
            //Matrix<float> trainingDataMat = new Matrix<float>(count, 2);
            //Matrix<float> trainingDataMat1 = trainingDataMat.GetRows(0, count >> 1, 1);
            //trainingDataMat1.SetRandNormal(new MCvScalar(200 / 512f), new MCvScalar(50 / 512f));
            //Matrix<float> trainingDataMat2 = trainingDataMat.GetRows(count >> 1, count, 1);
            //trainingDataMat2.SetRandNormal(new MCvScalar(300 / 512f), new MCvScalar(50 / 512f));

            TrainData tmpTrainData = new TrainData(trainingDataMat, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, labelsMats);
            bp.Train(tmpTrainData, (int)Emgu.CV.ML.MlEnum.AnnMlpTrainingFlag.Default);
//#if !NETFX_CORE
//                String fileName = Path.Combine(Application.StartupPath, "ann_mlp_model.xml");
//                bp.Save(fileName);
//                if (File.Exists(fileName))
//                    File.Delete(fileName);
//#endif
        }

        private void Form2_Load(object sender, EventArgs e)
        {
            CheckData();
            int width = 512, height = 512;
            Image<Bgr, Byte> img = new Image<Bgr, byte>(width, height);
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    Matrix<float> sampleMat = new Matrix<float>(1, 2);
                    sampleMat[0, 0] = i / (float)width;
                    sampleMat[0, 1] = j / (float)height;
                    //sampleMat[0, 0] = i;
                    //sampleMat[0, 1] = j;
                    //Mat sampleMat = new Mat(1, 2, Emgu.CV.CvEnum.DepthType.Cv32S, 1);
                    //sampleMat.SetTo<float>(new float[] { i / (float)width, j / (float)height });
                    Matrix<float> responseMat = new Matrix<float>(1, 1);
                    bp.Predict(sampleMat, responseMat);
                    if (responseMat[0, 0] >= 0.5)
                        img[i, j] = new Bgr(Color.Green.B, Color.Green.G, Color.Green.R);
                    else
                        img[i, j] = new Bgr(Color.Blue.B, Color.Blue.G, Color.Blue.R);
                }
            }
            pictureBox1.Image = img.Bitmap;
        }

        private void CheckData()
        {
            Matrix<float> sampleMat = new Matrix<float>(1, 2);
            Matrix<float> responseMat = new Matrix<float>(1, 1);
            //sampleMat[0, 0] = 1;
            //sampleMat[0, 1] = 2;
            sampleMat[0, 0] = 1 / (float)512;
            sampleMat[0, 1] = 2 / (float)512;
            bp.Predict(sampleMat, responseMat);
            float f = responseMat[0, 0];
            Console.WriteLine(String.Format("{0},{1}:{2}", sampleMat[0, 0], sampleMat[0, 1], responseMat[0, 0]));
            //sampleMat[0, 0] = 111;
            //sampleMat[0, 1] = 112;
            sampleMat[0, 0] = 111 / (float)512;
            sampleMat[0, 1] = 112 / (float)512;
            bp.Predict(sampleMat, responseMat);
            Console.WriteLine(String.Format("{0},{1}:{2}", sampleMat[0, 0], sampleMat[0, 1], responseMat[0, 0]));
            //sampleMat[0, 0] = 21;
            //sampleMat[0, 1] = 22;
            sampleMat[0, 0] = 21 / (float)512;
            sampleMat[0, 1] = 22 / (float)512;
            bp.Predict(sampleMat, responseMat);
            Console.WriteLine(String.Format("{0},{1}:{2}", sampleMat[0, 0], sampleMat[0, 1], responseMat[0, 0]));
        }
    }
}
