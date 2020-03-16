package at.ac.tuwien.caa.docscan.logic;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

import at.ac.tuwien.caa.docscan.camera.cv.DkPolyRect;
import at.ac.tuwien.caa.docscan.camera.cv.NativeWrapper;


public class IlluminationCorrection {

    private static IlluminationCorrection sInstance;
    private double[] correctionFactors;
    private boolean correctionFactorsSet = false;
    private static final double THRESHOLD = 50.0;

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
        List<Mat> channels = new ArrayList<>();
        Core.split(scaledMat, channels);

        //only green channel
        if(scaledMat.channels() == 3)
            scaledMat = desaturate(scaledMat);

        Mat red = channels.get(0);
        Mat blue = channels.get(2);

        double[] pixelsGreen= imageToDouble(scaledMat);
        double[] pixelRed = imageToDouble(red);
        double[] pixelBlue = imageToDouble(blue);
        for (int i = 0; i < pixelsGreen.length; i++) {
            if(pixelsGreen[i]>THRESHOLD || pixelRed[i]>THRESHOLD || pixelBlue[i]>THRESHOLD) {
                pixelRed[i] *= correctionFactors[i];
                pixelsGreen[i] *= correctionFactors[i];
                pixelBlue[i] *= correctionFactors[i];
            }
        }
        scaledMat.put(0,0,pixelsGreen);
        scaledMat = scaleImage(scaledMat, (double)1/3);
        scaledMat.convertTo(scaledMat, CvType.CV_8UC1);


        red.put(0,0, pixelRed);
        red = scaleImage(red, (double)1/3);
        red.convertTo(red, CvType.CV_8UC1);

        blue.put(0,0, pixelBlue);
        blue = scaleImage(blue, (double)1/3);
        blue.convertTo(blue, CvType.CV_8UC1);

        List<Mat> result = new ArrayList<>();
        result.add(red);
        result.add(scaledMat);
        result.add(blue);
        Mat outval = new Mat();
        Core.merge(result, outval);

        return convertMatToBitmapToByteArray(outval);
    }

    public void calculateCorrectionFactors(byte[] inputByteStream){
        correctionFactorsSet = true;
        Mat mat = Imgcodecs.imdecode(new MatOfByte(inputByteStream), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        Mat scaledMat = scaleImage(mat,3);
        //only green channel
        if(scaledMat.channels() == 3)
            scaledMat = desaturate(scaledMat);

        double[] pixels= imageToDouble(scaledMat);
        double maxValue = findMaximumIlluminationIntensityValue(pixels);
        correctionFactors = new double[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            correctionFactors[i] = maxValue/pixels[i];
        }
    }

    private double findMaximumIlluminationIntensityValue(double[] pixels){
        double maxValue = 0;
        for (int i = 0; i < pixels.length; i++) {
            if(pixels[i]> maxValue) {
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
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpeg", mat, matOfByte);
        return matOfByte.toArray();
    }

    public boolean isCorrectionFactorsSet(){
        return correctionFactorsSet;
    }

    private Mat desaturate(Mat mat){
        List<Mat> channels = new ArrayList<>();
        Core.split(mat, channels);
        //use only green channel in RGB images
        return channels.get(1);
    }

    /*    private byte[] illustrate(Mat mat, double[] correctionFactors){
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
    }*/

    private Mat performBinarization(Mat mat){
        Mat binarizedMat = new Mat();
        Imgproc.threshold(mat,binarizedMat,0,255,Imgproc.THRESH_OTSU);
        return binarizedMat;
    }
}