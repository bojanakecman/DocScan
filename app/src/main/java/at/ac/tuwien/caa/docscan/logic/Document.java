package at.ac.tuwien.caa.docscan.logic;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;

/**
 * Created by fabian on 01.02.2018.
 */

public class Document {

    private String mTitle;
    private ArrayList<Page> mPages;
    private boolean mIsUploaded = false;
    private boolean mIsCropped = false;
    private boolean mIsAwaitingUpload = false;
    private int mDpi;

    private TranskribusMetaData mMetaData = null;

    public Document() {

    }

    public Document(String title) {

        mTitle = title;
        mPages = new ArrayList<>();

    }

    public void validatePages() {

        if (mPages == null)
            return;

        Iterator<Page> it = mPages.iterator();
        while (it.hasNext()) {
            Page page = it.next();
            if (page == null || page.getFile() == null || !page.getFile().exists()) {
                it.remove();
            }
        }

    }

    public void replacePage(File file, int index) {

        if (mPages != null && mPages.size() >= index + 1 && mPages.get(index) != null) {

            Page page = mPages.get(index);
            page.getFile().delete();
            page.setFile(file);

            mIsUploaded = false;

        }

    }

    public void setMetaData(TranskribusMetaData metaData) {
        mMetaData = metaData;

    }

    public TranskribusMetaData getMetaData() {

        return mMetaData;

    }

    public ArrayList<String> getFileNames() {

        if (mPages == null)
            return null;

        ArrayList<String> fileNames = new ArrayList<>(mPages.size());
        for (Page page : mPages) {
            fileNames.add(page.getFile().getName());
        }

        return fileNames;

    }

    public ArrayList<File> getFiles() {

        if (mPages == null)
            return null;

        ArrayList<File> files = new ArrayList<>(mPages.size());
        for (Page page : mPages) {
            files.add(page.getFile());
        }

        return files;

    }

    public void setDir(File newDir) {

        mTitle = newDir.getName();

        for (Page page : mPages)
            page.changeDir(newDir);

    }


    public void setIsAwaitingUpload(boolean isAwaitingUpload) {

        mIsAwaitingUpload = isAwaitingUpload;

    }

    public boolean isAwaitingUpload() {

        return mIsAwaitingUpload;

    }

    public void setIsUploaded(boolean isUploaded) {

        mIsUploaded = isUploaded;

    }

    public boolean isUploaded() {

        return mIsUploaded;

    }

    public void setIsCropped(boolean isCropped) {

        mIsCropped = isCropped;

    }

    public boolean isCropped() {

        return mIsCropped;

    }

    public String getTitle() {
        return mTitle;

    }

    public void setTitle(String title) {

        mTitle = title;

    }

    public ArrayList<Page> getPages() {

        return mPages;

    }

    public void setPages(ArrayList<Page> pages) {

        mPages = pages;

    }

    public int getDpi(){
        return mDpi;
    }

    public void setDpi(int dpi){
        mDpi = dpi;
        storeDpiToMetadata();
    }

    private void storeDpiToMetadata(){
        if(mMetaData == null)
            mMetaData = new TranskribusMetaData();
        mMetaData.setDpi(mDpi);
        setMetaData(mMetaData);
    }
}
