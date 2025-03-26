import cv2


def read_avi(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Could not open file")
    else:
        print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("FPS       :", cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(filename.capitalize(), frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_avi('test.mp4')
