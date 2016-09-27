package at.ac.tuwien.caa.docscan.cv;

import android.graphics.PointF;

/**
 * Created by fabian on 27.09.2016.
 */
public class DkVector {

    public float x, y;

    public DkVector(PointF point1, PointF point2) {

        x = point1.x - point2.x;
        y = point1.y - point2.y;

    }

    public float scalarProduct(DkVector vector) {

        return x * vector.x + y * vector.y;

    }

}