package at.ac.tuwien.caa.docscan.logic;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IlluminationCorrection {

    private static IlluminationCorrection sInstance;
    private double[] correctionFactors;
    private Mat LUT;
    private boolean correctionFactorsSet = false;

    public static boolean isInstanceNull() {
        return sInstance == null;
    }

    public static IlluminationCorrection getInstance(){
        if (sInstance == null)
            sInstance = new IlluminationCorrection();

        return sInstance;
    }

    public byte[] antivignetting(byte[] inputByteStream){
        Mat mat = Imgcodecs.imdecode(new MatOfByte(inputByteStream), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        Mat scaledMat = scaleImage(mat,3);
        Mat correctedMat = new Mat();
        List<Mat> channels = new ArrayList<>();
        Core.split(scaledMat, channels);
        Mat red = channels.get(0);
        Mat green = channels.get(1);
        Mat blue = channels.get(2);

        double[] pixels= imageToDouble(green);
        double[] pixelsRed= imageToDouble(red);
        double[] pixelsBlue= imageToDouble(blue);
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] *= correctionFactors[i];

/*          pixelsRed[i] *= correctionFactors[i];
            pixelsBlue[i] *= correctionFactors[i];*/
        }
        green.put(0,0,pixels);

        red.convertTo(red, CvType.CV_32FC3);
        green.convertTo(green, CvType.CV_32FC3);
        blue.convertTo(blue, CvType.CV_32FC3);


/*        Core.merge(Arrays.asList(red, green, blue), correctedMat);
        correctedMat.convertTo(correctedMat, CvType.CV_8UC3);*/

        green.convertTo(green, CvType.CV_8UC1);
        green = scaleImage(green, (double)1/3);

        return convertMatToBitmapToByteArray(green);
    }

    public byte[] calculateCorrectionFactors(byte[] inputByteStream){
        //only green channel
        correctionFactorsSet = true;
        Mat mat = Imgcodecs.imdecode(new MatOfByte(inputByteStream), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        Mat scaledMat = scaleImage(mat,3);
        LUT = new Mat(scaledMat.rows(), scaledMat.cols(), CvType.CV_8UC1);
        List<Mat> channels = new ArrayList<>();
        Core.split(scaledMat, channels);
        Mat greenChannel = channels.get(1);
        double[] pixels= imageToDouble(greenChannel);
        double maxValue = findMaximumIlluminationIntensityValue(pixels);
        correctionFactors = new double[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            correctionFactors[i] = maxValue/pixels[i];
        }

        LUT.put(0,0,correctionFactors);
        LUT = scaleImage(LUT, (double)1/3);
        //byte[] byteArray = illustrate(scaledMat, correctionFactors);

/*        greenChannel.put(0,0,correctionFactors);
        greenChannel.convertTo(greenChannel, CvType.CV_8UC1);
        greenChannel = scaleImage(greenChannel, (double)1/3);
        byte[] byteArray = convertMatToBitmapToByteArray(greenChannel);*/

        return  convertMatToBitmapToByteArray(LUT);

    }

    private double findMaximumIlluminationIntensityValue(double[] pixels){
        double maxValue = 0;
        for (int i = 0; i < pixels.length; i++) {
            if(pixels[i]> maxValue){
                maxValue = pixels[i];
            }
        }
        return maxValue;
    }

    private Mat scaleImage(Mat mat, double scalingFactor){
        Mat scaledMat = new Mat();
        //Scaling the Image
        Imgproc.resize(mat, scaledMat, new Size(mat.cols()/scalingFactor, mat.rows()/scalingFactor), 0, 0,
                Imgproc.INTER_AREA);
        return scaledMat;
    }


    private double[] imageToDouble(Mat mat){
        mat.convertTo(mat, CvType.CV_64FC1);
        int size = (int) (mat.total() * mat.channels());
        double[] pixels = new double[size];
        mat.get(0, 0, pixels);
        return pixels;
    }

    private byte[] convertMatToBitmapToByteArray(Mat mat){
        Bitmap image = Bitmap.createBitmap(mat.cols(),
                mat.rows(), Bitmap.Config.RGB_565);

        Utils.matToBitmap(mat, image);
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        return byteArrayOutputStream .toByteArray();
    }


    private byte[] illustrate(Mat mat, double[] correctionFactors){
        double max = 0.0;
        double [] pixels = new double[correctionFactors.length];
        for (int i = 0; i < correctionFactors.length; i++) {
            if(correctionFactors[i]> max){
                max = correctionFactors[i];
            }
        }
        double difference = max - 1;
        for(int i = 0; i < correctionFactors.length; i++){
            double value = ((correctionFactors[i] - 1)/difference)*255;
            pixels[i] = value;
        }

        Mat illustrateMat = new Mat(mat.rows(), mat.cols(), CvType.CV_8UC1);
        illustrateMat.put(0,0, pixels);

        illustrateMat = scaleImage(illustrateMat, (double)1/3);
        return  convertMatToBitmapToByteArray(illustrateMat);

    }

    public boolean isCorrectionFactorsSet(){
        return correctionFactorsSet;
    }
}