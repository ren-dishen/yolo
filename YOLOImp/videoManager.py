import time
import os
import cv2

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """

    
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break


def processImagesParallel(sess, input, outputs):
    from joblib import Parallel, delayed
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    images = [img for img in os.listdir(input) if img.endswith(".jpg")]
    
    count = 0
    newImages = list(images)
    for image in images:
        newImages[count] = os.path.join(input, image)
        count += 1

    Parallel(n_jobs=num_cores, backend="threading")(delayed(predict)(sess, os.path.join(input, image)) for image in images)

    
def processImages(sess, input, output):
    images = [img for img in os.listdir(input) if img.endswith(".jpg")]
    for image in images:
        predict(sess, os.path.join(input, image))

def framesToVideo(input, ouput):

    image_folder = input
    video_name = ouput

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    imgName = os.path.join(image_folder, images[0])
    frame = cv2.imread(imgName)
    
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, fourcc, 24, (width,height), True)
    count = 0
    for image in images:
        count += 1
        imageName = os.path.join(image_folder, image)
        print(count)
        video.write(cv2.imread(imageName))
        

    cv2.destroyAllWindows()
    video.release()