using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.ML;
using System.Reflection;
using System.Threading;

namespace EmguCVDemo
{
    public partial class Form1 : Form
    {
        VideoCapture cap;
        //CascadeClassifier cascadeClassifier;
        Mat nextFrame = new Mat();
        Bitmap nowImg;
        //SVM svmObj = new SVM();
        Rectangle faceRectangle = new Rectangle();
        Rectangle faceReadRectangle = new Rectangle();
        string url = "";
        //string url = "rtsp://admin:admin123@192.168.3.64:554/h264/ch1/main/av_stream";
        //string url = "rtsp://admin:admin@192.168.0.199/0";
        public Form1()
        {
            InitializeComponent();
            CreateBP();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            //CvInvoke.UseOpenCL = false;
            // passing 0 gets zeroth webcam
            cap = new VideoCapture(url);
            cap.ImageGrabbed += new EventHandler(cap_ImageGrabbed);
            // adjust path to find your xml
            //cascadeClassifier = new CascadeClassifier("haarcascade_frontalface_alt2.xml");

            btnStart.Enabled = true;
            btnPause.Enabled = false;
            btnStop.Enabled = false;
        }

        private void cap_ImageGrabbed(object sender, EventArgs e)
        {
            GetFrame();
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            cap.Start();
            btnStart.Enabled = false;
            btnPause.Enabled = true;
            btnStop.Enabled = true;
        }

        private void btnPause_Click(object sender, EventArgs e)
        {
            cap.Pause();
            btnStart.Enabled = true;
            btnPause.Enabled = false;
            btnStop.Enabled = false;
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            cap.Stop();
            //cap.Dispose();
            //cap = null;
            //GC.Collect();
            //cap = new VideoCapture(url);
            //cap.ImageGrabbed += new EventHandler(cap_ImageGrabbed);
            btnStart.Enabled = true;
            btnPause.Enabled = false;
            btnStop.Enabled = false;
            //Console.WriteLine(String.Format("start:{0:yyyy-MM-dd HH:mm:ss.fff}", DateTime.Now));
            //Image<Bgr, Byte> img = new Image<Bgr, Byte>(nowImg);
            //// there's only one channel (greyscale), hence the zero index
            //var faces = cascadeClassifier.DetectMultiScale(
            //    img, 1.1, 3,
            //    new Size(80, 80),
            //    new Size(nextFrame.Width / 8, nextFrame.Height / 8)
            //    );

            //foreach (var face in faces)
            //{
            //    img.Draw(face, new Bgr(0, double.MaxValue, 0), 3);
            //}
            //pbImg.Image = nextFrame.Bitmap;
            nowImg = (Bitmap)pbImg.Image;
            //Console.WriteLine(String.Format("end:{0:yyyy-MM-dd HH:mm:ss.fff}", DateTime.Now));
        }

        private void btnTrain_Click(object sender, EventArgs e)
        {
        }

        private bool isDown = false;
        private void pbImg_MouseDown(object sender, MouseEventArgs e)
        {
            //获取时间图片大小
            PropertyInfo rectangleProperty = this.pbImg.GetType().GetProperty("ImageRectangle", BindingFlags.Instance | BindingFlags.NonPublic);
            Rectangle rectangle = (Rectangle)rectangleProperty.GetValue(this.pbImg, null);
            DrawRectangle(e.X - rectangle.X, e.Y - rectangle.Y);
            faceRectangle.X = e.X - rectangle.X;
            faceRectangle.Y = e.Y - rectangle.Y;
            //faceRectangle.Width = 0;
            //faceRectangle.Height = 0;
            DrawRectangle(e.X - rectangle.X, e.Y - rectangle.Y);
            isDown = true;
            PrintMsg(String.Format("MouseDown:{0},{1}", e.X - rectangle.X, e.Y - rectangle.Y));
        }

        private void pbImg_MouseMove(object sender, MouseEventArgs e)
        {
            if (!isDown)
                return;
            //获取时间图片大小
            PropertyInfo rectangleProperty = this.pbImg.GetType().GetProperty("ImageRectangle", BindingFlags.Instance | BindingFlags.NonPublic);
            Rectangle rectangle = (Rectangle)rectangleProperty.GetValue(this.pbImg, null);
            DrawRectangle(e.X - rectangle.X, e.Y - rectangle.Y);
        }

        private void pbImg_MouseUp(object sender, MouseEventArgs e)
        {
            isDown = false;
            //获取时间图片大小
            PropertyInfo rectangleProperty = this.pbImg.GetType().GetProperty("ImageRectangle", BindingFlags.Instance | BindingFlags.NonPublic);
            Rectangle rectangle = (Rectangle)rectangleProperty.GetValue(this.pbImg, null);
            DrawRectangle(e.X - rectangle.X, e.Y - rectangle.Y);
            if (faceRectangle.Width > 0 && faceRectangle.Height > 0)
                pbSubImg.Image = new Bitmap(nowImg).Clone(faceReadRectangle, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            PrintMsg(String.Format("MouseUp:{0},{1}", e.X - rectangle.X, e.Y - rectangle.Y));
        }

        private void DrawRectangle(int x, int y)
        {
            if (nowImg == null || btnStop.Enabled)
                return;
            //获取实际图片大小
            PropertyInfo rectangleProperty = this.pbImg.GetType().GetProperty("ImageRectangle", BindingFlags.Instance | BindingFlags.NonPublic);
            Rectangle rectangle = (Rectangle)rectangleProperty.GetValue(this.pbImg, null);
            int w = Math.Abs(x - faceRectangle.X), h = Math.Abs(y - faceRectangle.Y);
            if (faceRectangle.X > x)
            {
                faceRectangle.X = x;
            }
            if (faceRectangle.Y > y)
            {
                faceRectangle.Y = y;
            }
            faceRectangle.Width = w;
            faceRectangle.Height = h;
            double wb = rectangle.Width / (double)nowImg.Width;
            double hb = rectangle.Height / (double)nowImg.Height;
            faceReadRectangle = new Rectangle(faceRectangle.Location, faceRectangle.Size);
            faceReadRectangle.X = Convert.ToInt32(faceRectangle.X / wb);
            faceReadRectangle.Y = Convert.ToInt32(faceRectangle.Y / hb);
            faceReadRectangle.Width = Convert.ToInt32(faceRectangle.Width / wb);
            faceReadRectangle.Height = Convert.ToInt32(faceRectangle.Height / hb);

            Bitmap cloneImg = new Bitmap(nowImg);
            Graphics g = Graphics.FromImage(cloneImg);
            int lineWide = Convert.ToInt32(2.0 / pbImg.Width * cloneImg.Width);
            g.DrawRectangle(new Pen(Color.Red, lineWide), faceReadRectangle);
            g.Dispose();
            pbImg.Image = cloneImg;
            GC.Collect();
        }

        private void PrintMsg(string msg)
        {
            txtMsg.Invoke(new Action(() =>
            {
                txtMsg.AppendText(msg + "\r\n");
            }));
        }
        Object lockObj = new Object();
        private void GetFrame()
        {
            try
            {
                lock (lockObj)
                {
                    if (cap != null && cap.Ptr != IntPtr.Zero)
                    {
                        if (!cap.IsOpened)
                            return;
                        cap.Retrieve(nextFrame, 0);
                        if (!nextFrame.IsEmpty)
                        {
                            pbImg.Image = nextFrame.Bitmap;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }
        }

        private void btnPredict_Click(object sender, EventArgs e)
        {
            Rectangle[] rects = PredictBP(nowImg);
            Bitmap cloneImg = new Bitmap(nowImg);
            Graphics g = Graphics.FromImage(cloneImg);
            int lineWide = Convert.ToInt32(2.0 / pbImg.Width * cloneImg.Width);
            foreach (var r in rects)
            {
                var tmpR = r;
                tmpR.X = (int)(tmpR.X / (float)bpWidth * cloneImg.Width);
                tmpR.Y = (int)(tmpR.Y / (float)bpHeight * cloneImg.Height);
                tmpR.Width = (int)(tmpR.Width / (float)bpWidth * cloneImg.Width);
                tmpR.Height = (int)(tmpR.Height / (float)bpHeight * cloneImg.Height);
                g.DrawRectangle(new Pen(Color.Red, lineWide), tmpR);
            }
            g.Dispose();
            pbImg.Image = cloneImg;
            MessageBox.Show("识别完成");
        }
        #region BP神经网络
        private int bpWidth = 960, bpHeight = 540;
        //private int bpWidth = 1920, bpHeight = 1080;
        private int bpRectangleCount = 50;
        private ANN_MLP bp;
        private void CreateBP()
        {
            bp = new ANN_MLP();
            Matrix<int> layerSizes = new Matrix<int>(new int[] { 
                bpWidth * bpHeight * 3,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                bpRectangleCount * 4 
            });
            bp.SetLayerSizes(layerSizes);
            bp.SetActivationFunction(ANN_MLP.AnnMlpActivationFunction.SigmoidSym, 1, 1);
            bp.TermCriteria = new MCvTermCriteria(10000, 1.0e-8);
            //bp.BackpropWeightScale = 0.1;
            //bp.BackpropMomentumScale = 0.1;
            bp.SetTrainMethod(ANN_MLP.AnnMlpTrainMethod.Backprop, 0.1, 0.1);
        }
        private void TrainBP(Bitmap img, Rectangle[] rects)
        {
            Bitmap tmpImg = ZoomImg(img, bpWidth, bpHeight, ref rects);
            Image<Bgr, float> trainingData = new Image<Bgr, float>(tmpImg);
            Matrix<float> trainingDataMats = new Matrix<float>(1, bpWidth * bpHeight * 3);
            trainingDataMats.Bytes = trainingData.Bytes;
            Matrix<float> labelsMats = new Matrix<float>(1, bpRectangleCount * 4);
            labelsMats.SetValue(-1);
            for (int i = 0; i < rects.Length * 4 && i < bpRectangleCount * 4; i += 4)
            {
                labelsMats[0, i] = rects[i / 4].X / (float)bpWidth;
                labelsMats[0, i + 1] = rects[i / 4].Y / (float)bpHeight;
                labelsMats[0, i + 2] = rects[i / 4].Width / (float)bpWidth;
                labelsMats[0, i + 3] = rects[i / 4].Height / (float)bpHeight;
            }
            TrainData tmpTrainData = new TrainData(trainingDataMats, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, labelsMats);
            bp.Train(tmpTrainData, (int)Emgu.CV.ML.MlEnum.AnnMlpTrainingFlag.Default);
            tmpImg.Dispose();
            tmpImg = null;
        }
        private Rectangle[] PredictBP(Bitmap img)
        {
            Rectangle[] tmpR = new Rectangle[] { };
            Bitmap tmpImg = ZoomImg(img, bpWidth, bpHeight, ref tmpR);
            List<Rectangle> retRects = new List<Rectangle>();
            Image<Bgr, float> trainingData = new Image<Bgr, float>(img);
            Matrix<float> trainingDataMats = new Matrix<float>(1, bpWidth * bpHeight * 3);
            trainingDataMats.Bytes = trainingData.Bytes;
            Matrix<float> labelsMats = new Matrix<float>(1, bpRectangleCount * 4);
            bp.Predict(trainingDataMats, labelsMats);
            for (int i = 0; i < bpRectangleCount * 4; i += 4)
            {
                Rectangle r = new Rectangle();
                r.X = (int)(labelsMats[0, i] * (float)bpWidth);
                r.Y = (int)(labelsMats[0, i + 1] * (float)bpHeight);
                r.Width = (int)(labelsMats[0, i + 2] * (float)bpWidth);
                r.Height = (int)(labelsMats[0, i + 3] * (float)bpHeight);
                if (r.X >= 0 && r.Y >= 0 && r.Width > 0 && r.Height > 0)
                    retRects.Add(r);
            }
            return retRects.ToArray();
        }

        private Bitmap ZoomImg(Bitmap img, int width, int height, ref Rectangle[] rects)
        {
            //缩放图片
            Bitmap tmpZoomImg = new Bitmap(width, height);
            Graphics g = Graphics.FromImage(tmpZoomImg);
            g.Clear(Color.Black);
            int x = 0, y = 0, w = width, h = height;
            double b1, b2, b0;
            b1 = width / height;
            b2 = img.Width / img.Height;
            if (b1 > b2)
            {
                w = (int)(h * b2);
                x = (width - w) / 2;
            }
            else if (b2 > b1)
            {
                h = (int)(w / b2);
                y = (height - h) / 2;
            }
            g.DrawImage(img, x, y, w, h);
            g.Dispose();
            //转换矩形比例
            for (int i = 0; i < rects.Length; i++)
            {
                Rectangle rectangle = new Rectangle();
                double wb = img.Width / (double)width;
                double hb = img.Height / (double)height;
                rectangle.X = Convert.ToInt32(rects[i].X / wb);
                rectangle.Y = Convert.ToInt32(rects[i].Y / hb);
                rectangle.Width = Convert.ToInt32(rects[i].Width / wb);
                rectangle.Height = Convert.ToInt32(rects[i].Height / hb);
                rects[i] = rectangle;
            }
            return tmpZoomImg;
        }
        #endregion

        private void btnImport_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                nowImg = (Bitmap)Image.FromFile(ofd.FileName);
                pbImg.Image = nowImg;
                MessageBox.Show("导入完成");
            }
        }

        private void btnExport_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                nowImg.Save(sfd.FileName);
                MessageBox.Show("导出完成");
            }
        }

        private void btn_Click(object sender, EventArgs e)
        {
            TrainBP(nowImg, new Rectangle[] { faceReadRectangle });
            MessageBox.Show("训练完成");
        }

        private void btnSaveBP_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "XML文件(*.xml)|*.xml";
            if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                //bp.Save(sfd.FileName);
                bp.Write(new FileStorage(sfd.FileName, FileStorage.Mode.Write));
                MessageBox.Show("导出完成");
            }
        }

        private void btnLoadBP_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "XML文件(*.xml)|*.xml";
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                bp.Read(new FileStorage(ofd.FileName, FileStorage.Mode.Read).GetRoot());
                MessageBox.Show("导入完成");
            }
        }
    }
}
