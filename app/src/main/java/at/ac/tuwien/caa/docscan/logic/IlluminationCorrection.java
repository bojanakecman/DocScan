package at.ac.tuwien.caa.docscan.logic;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IlluminationCorrection {

    private static IlluminationCorrection sInstance;
    private Mat correctionFactors;
    private boolean correctionFactorsSet = false;
    private static int KERNEL_SIZE = 31;

    public static boolean isInstanceNull() {
        return sInstance == null;
    }

    public static IlluminationCorrection getInstance(){
        if (sInstance == null)
            sInstance = new IlluminationCorrection();
        return sInstance;
    }

    public byte[] antivignetting(byte[] inputByteStream){
        Mat jpegData = new Mat(1, inputByteStream.length, CvType.CV_8UC1);
        jpegData.put(0, 0, inputByteStream);
        Mat mat = Imgcodecs.imdecode(jpegData, Imgcodecs.IMREAD_COLOR);
        mat.convertTo(mat, CvType.CV_32FC3);

        List<Mat> list = Arrays.asList( this.correctionFactors, this.correctionFactors, this.correctionFactors);
        List<Mat> correctionFactorList = new ArrayList<>(list);

        Mat correctionFactorsRGB = new Mat();
        Core.merge(correctionFactorList, correctionFactorsRGB);

        Core.multiply(correctionFactorsRGB, mat, mat);

        mat.convertTo(mat, CvType.CV_8UC3);

        return convertMatToBitmapToByteArray(mat);
    }

    public void calculateCorrectionFactors(byte[] inputByteStream){
        Mat mat = Imgcodecs.imdecode(new MatOfByte(inputByteStream), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);

        //only green channel
        if(mat.channels() == 3)
            mat = desaturate(mat);

        Imgproc.medianBlur(mat, mat, KERNEL_SIZE);

        Core.MinMaxLocResult result = Core.minMaxLoc(mat);
        double maxValue = result.maxVal;

        this.correctionFactors = Mat.zeros(mat.rows(), mat.cols(), CvType.CV_32FC1);

        Mat helpMat = Mat.ones(mat.rows(), mat.cols(), mat.type());

        Core.divide(helpMat, mat, correctionFactors, maxValue, CvType.CV_32FC1);

        correctionFactorsSet = true;
    }

    private byte[] convertMatToBitmapToByteArray(Mat mat){
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, matOfByte);
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
}