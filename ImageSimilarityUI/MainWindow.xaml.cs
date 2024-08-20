using OpenCvSharp;
using System;
using System.Collections.ObjectModel;
using System.Data;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Path = System.IO.Path;
using Size = OpenCvSharp.Size;
using Window = System.Windows.Window;

namespace ImageSimilarityUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private List<string> _imageFiles;
        public MainWindow()
        {
            InitializeComponent();
            _imageFiles = Directory.GetFiles(Directory.GetCurrentDirectory(), "images/*.png").ToList(); // 只加载jpg格式，可以扩展支持其他格式
            InitializeGrid();
        }
        private void InitializeGrid()
        {
            int rowCount = _imageFiles.Count + 1; // N rows
            int columnCount = Enum.GetValues(typeof(AlgorithmType)).Length+1; // M columns

            SetupGrid(rowCount, columnCount);

            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        var content = new CellContent
                        {
                            Text = @"imgae\image"
                        };
                        AddCellContent(i, j, content);
                    }
                    else
                    {
                        if (i == 0)
                        {
                            //列
                            var content = new CellContent
                            {
                                Text = $"{Path.GetFileName(_imageFiles[0])}  {((AlgorithmType)(j - 1)).ToString()}算法",
                                ImageUrl = _imageFiles[0] // 偶数列显示图片
                            };
                            AddCellContent(i, j, content);
                        }
                        else
                        {
                            //第二行开始,第一列为image和名字
                            if (j == 0)
                            {
                                var content = new CellContent
                                {
                                    Text = $"{Path.GetFileName(_imageFiles[i - 1])}",
                                    ImageUrl = _imageFiles[i - 1] // 偶数列显示图片
                                };
                                AddCellContent(i, j, content);
                            }
                            else
                            {
                                string similarity = CompareTwoImages(_imageFiles[i - 1], _imageFiles[0], (AlgorithmType)(j - 1));
                                var content = new CellContent
                                {
                                    Text = similarity,
                                };
                                AddCellContent(i, j, content);
                            }
                        }
                    }

                }
            }
        }

        private void SetupGrid(int rowCount, int columnCount)
        {
            DynamicGrid.RowDefinitions.Clear();
            DynamicGrid.ColumnDefinitions.Clear();

            for (int i = 0; i < rowCount; i++)
            {
                DynamicGrid.RowDefinitions.Add(new RowDefinition());
            }

            for (int j = 0; j < columnCount; j++)
            {
                DynamicGrid.ColumnDefinitions.Add(new ColumnDefinition());
            }
        }
        private void AddCellContent(int row, int column, CellContent content)
        {
            var stackPanel = new StackPanel
            {
                Orientation = Orientation.Vertical,
                VerticalAlignment = VerticalAlignment.Center,
                HorizontalAlignment = HorizontalAlignment.Center
            };

            if (!string.IsNullOrEmpty(content.ImageUrl))
            {
                var image = new Image
                {
                    Source = new BitmapImage(new Uri(content.ImageUrl, UriKind.RelativeOrAbsolute)),
                    Width = 200,
                    Height = 150,
                    Margin = new Thickness(2)
                };
                stackPanel.Children.Add(image);
            }

            var textBlock = new TextBlock
            {
                Text = content.Text,
                VerticalAlignment = VerticalAlignment.Bottom
            };
            stackPanel.Children.Add(textBlock);

            Grid.SetRow(stackPanel, row);
            Grid.SetColumn(stackPanel, column);
            DynamicGrid.Children.Add(stackPanel);
        }

        private string CompareTwoImages(string img1Path, string img2Path, AlgorithmType algorithmType)
        {
            Mat img1 = Cv2.ImRead(img1Path, ImreadModes.Grayscale);
            Mat img2 = Cv2.ImRead(img2Path, ImreadModes.Grayscale);

            // 调整图像尺寸相同
            Mat resizedImg2 = new Mat();
            Cv2.Resize(img2, resizedImg2, new Size(img1.Width, img1.Height));

            switch (algorithmType)
            {
                case AlgorithmType.Perceptual_PHash:
                    return CalculatePHash(img1, resizedImg2);
                case AlgorithmType.SSIM:
                    return CalculateSSIM(img1, resizedImg2);
                //case AlgorithmType.Histogram:
                //    return CalculateHistogram(img1, resizedImg2);
            }
            return string.Empty;
        }

        #region 感知哈希（Perceptual Hash, pHash）

        private string CalculatePHash(Mat img1, Mat img2)
        {
            ulong hash1 = ComputePhash(img1);
            ulong hash2 = ComputePhash(img2);
            return ComputeSimilarityPercentage(hash1, hash2);
        }

        private ulong ComputePhash(Mat img)
        {
            Mat resized = new Mat();
            Cv2.Resize(img, resized, new Size(32, 32));
            Mat dct = new Mat();
            var outMat = new Mat();
            resized.ConvertTo(outMat, MatType.CV_32F);
            Cv2.Dct(outMat, dct);
            Mat topLeft = dct.ColRange(0, 8).RowRange(0, 8);
            double mean = Cv2.Mean(topLeft).Val0;

            ulong hash = 0;
            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 8; x++)
                {
                    hash <<= 1;
                    if (topLeft.At<float>(y, x) > mean)
                        hash |= 1;
                }
            }
            return hash;
        }
        private string ComputeSimilarityPercentage(ulong hash1, ulong hash2)
        {
            int distance = HammingDistance(hash1, hash2);
            int totalBits = 64;
            return $"{(1 - (double)distance / totalBits) * 100:F2}%";
        }
        private int HammingDistance(ulong hash1, ulong hash2)
        {
            ulong x = hash1 ^ hash2;
            int setBits = 0;
            while (x > 0)
            {
                setBits += (int)(x & 1);
                x >>= 1;
            }
            return setBits;
        }
        #endregion

        #region SSIM
        private string CalculateSSIM(Mat img1, Mat img2)
        {
            const double C1 = 6.5025;
            const double C2 = 58.5225;

            // 将输入图像转换为 CV_64F 类型
            Mat img1_64F = new Mat();
            Mat img2_64F = new Mat();
            img1.ConvertTo(img1_64F, MatType.CV_64F);
            img2.ConvertTo(img2_64F, MatType.CV_64F);

            Mat img1Squared = img1_64F.Mul(img1_64F);
            Mat img2Squared = img2_64F.Mul(img2_64F);
            Mat img1Img2 = img1_64F.Mul(img2_64F);

            Mat mu1 = new Mat();
            Mat mu2 = new Mat();

            Cv2.GaussianBlur(img1_64F, mu1, new Size(11, 11), 1.5);
            Cv2.GaussianBlur(img2_64F, mu2, new Size(11, 11), 1.5);

            Mat mu1Squared = mu1.Mul(mu1);
            Mat mu2Squared = mu2.Mul(mu2);
            Mat mu1Mu2 = mu1.Mul(mu2);

            Mat sigma1Squared = new Mat();
            Mat sigma2Squared = new Mat();
            Mat sigma12 = new Mat();

            Cv2.GaussianBlur(img1Squared, sigma1Squared, new Size(11, 11), 1.5);
            Cv2.GaussianBlur(img2Squared, sigma2Squared, new Size(11, 11), 1.5);
            Cv2.GaussianBlur(img1Img2, sigma12, new Size(11, 11), 1.5);

            sigma1Squared -= mu1Squared;
            sigma2Squared -= mu2Squared;
            sigma12 -= mu1Mu2;

            // 将C1和C2转换为Mat类型
            Mat C1Mat = new Mat(mu1.Size(), MatType.CV_64F, new Scalar(C1));
            Mat C2Mat = new Mat(mu1.Size(), MatType.CV_64F, new Scalar(C2));

            Mat t1 = 2 * mu1Mu2 + C1Mat;
            Mat t2 = 2 * sigma12 + C2Mat;
            Mat t3 = t1.Mul(t2);

            t1 = mu1Squared + mu2Squared + C1Mat;
            t2 = sigma1Squared + sigma2Squared + C2Mat;
            t1 = t1.Mul(t2);

            Mat ssimMap = new Mat();
            Cv2.Divide(t3, t1, ssimMap);

            Scalar mssim = Cv2.Mean(ssimMap);
            return $"{mssim.Val0 * 100:F2}%" ;
        }
        #endregion

        #region 直方图

        private string CalculateHistogram(Mat img1, Mat img2) 
        {
            // 确保图像为3通道
            img1 = EnsureThreeChannels(img1);
            img2 = EnsureThreeChannels(img2);

            // 将图像转换为 HSV 色彩空间
            Mat hsvImg1 = new Mat();
            Mat hsvImg2 = new Mat();
            Cv2.CvtColor(img1, hsvImg1, ColorConversionCodes.BGR2HSV);
            Cv2.CvtColor(img2, hsvImg2, ColorConversionCodes.BGR2HSV);

            // 分通道计算直方图
            Mat hist1_H = CalculateHistogram(hsvImg1, 0); // H通道
            Mat hist1_S = CalculateHistogram(hsvImg1, 1); // S通道
            Mat hist2_H = CalculateHistogram(hsvImg2, 0);
            Mat hist2_S = CalculateHistogram(hsvImg2, 1);

            // 使用H通道和S通道进行比较
            double correlation_H = Cv2.CompareHist(hist1_H, hist2_H, HistCompMethods.Correl);
            double correlation_S = Cv2.CompareHist(hist1_S, hist2_S, HistCompMethods.Correl);

            // 取均值作为最终的相似度
            double avgCorrelation = (correlation_H + correlation_S) / 2;
            double similarityPercentage = avgCorrelation * 100;

            // 输出结果
            Console.WriteLine($"Histogram Similarity: {similarityPercentage:F2}%");

            return $"Histogram Similarity: {similarityPercentage:F2}%";

            //// 计算直方图
            //Mat histImg1 = CalculateHistogram(hsvImg1);
            //Mat histImg2 = CalculateHistogram(hsvImg2);

            //// 直方图比较并转换为百分比
            //double correlation = Cv2.CompareHist(histImg1, histImg2, HistCompMethods.Correl);
            //double chiSquare = Cv2.CompareHist(histImg1, histImg2, HistCompMethods.Chisqr);
            //double intersection = Cv2.CompareHist(histImg1, histImg2, HistCompMethods.Intersect);
            //double bhattacharyya = Cv2.CompareHist(histImg1, histImg2, HistCompMethods.Bhattacharyya);

            //// 将结果转换为相似性百分比
            //double correlationPercentage = (correlation + 1) / 2 * 100;  // 相关性 [-1, 1] 转换到 [0, 100]
            //double chiSquarePercentage = Math.Max(0, (1 - chiSquare / 1000)) * 100; // 卡方距离较小越相似，1000是一个假定的正常化因子
            //double intersectionPercentage = intersection * 100; // 交叉本身就是 [0, 1] 范围内的值
            //double bhattacharyyaPercentage = (1 - bhattacharyya) * 100; // 巴氏距离越小越相似

            //// 输出结果
            //Console.WriteLine($"Correlation Similarity: {correlationPercentage:F2}%");
            //Console.WriteLine($"Chi-Square Similarity: {chiSquarePercentage:F2}%");
            //Console.WriteLine($"Intersection Similarity: {intersectionPercentage:F2}%");
            //Console.WriteLine($"Bhattacharyya Similarity: {bhattacharyyaPercentage:F2}%");

            //return $"相关性 Similarity: {correlationPercentage:F2}% \n" + $"卡方距离 Similarity: {chiSquarePercentage:F2}% \n"
            //    + $"交叉 Similarity: {intersectionPercentage:F2}%\n" + $"巴氏距离 Similarity: {bhattacharyyaPercentage:F2}%\n";
        }

        private Mat CalculateHistogram(Mat hsvImage)
        {
            // 调整直方图的参数
            int[] histSize = { 50, 60 }; // 使用较高分辨率的直方图
            Rangef[] ranges = { new Rangef(0, 180), new Rangef(0, 256) }; // H通道范围(0-180), S通道范围(0-256)
            int[] channels = { 0, 1 }; // 使用H通道和S通道

            // 计算直方图
            Mat hist = new Mat();
            Cv2.CalcHist(new Mat[] { hsvImage }, channels, null, hist, 2, histSize, ranges, true, false);

             Cv2.Normalize(hist, hist, 0, 1, NormTypes.MinMax);

            return hist;
        }

        static Mat CalculateHistogram(Mat image, int channel)
        {
            int[] histSize = { 256 }; // 256个箱子
            Rangef[] ranges = { new Rangef(0, 256) }; // 范围0到255

            // 计算直方图
            Mat hist = new Mat();
            Cv2.CalcHist(new Mat[] { image }, new int[] { channel }, null, hist, 1, histSize, ranges);

            // 归一化直方图
            Cv2.Normalize(hist, hist, 0, 1, NormTypes.MinMax);

            return hist;
        }

        // 确保图像为3通道的函数
        private Mat EnsureThreeChannels(Mat img)
        {
            if (img.Channels() == 1)
            {
                Mat img3Channel = new Mat();
                Cv2.CvtColor(img, img3Channel, ColorConversionCodes.GRAY2BGR);
                return img3Channel;
            }
            return img;
        }

        #endregion

    }
    public class CellContent
    {
        public string Text { get; set; }
        public string ImageUrl { get; set; } // 如果没有图片，则为null
    }

    public enum AlgorithmType
    {
        Perceptual_PHash,
        SSIM,
        //Histogram
    }
}

