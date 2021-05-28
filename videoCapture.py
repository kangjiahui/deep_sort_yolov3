import cv2
from multiprocessing import Process, Queue


def tradition_method(_cap):
    while _cap.isOpened():
        ret, frame = _cap.read()
        cv2.namedWindow("test", 0)
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def receive_data(queue):
    rtsp = "rtsp://admin:bocom123456@10.20.40.205:554"
    flv1 = "http://10.20.40.44:8080/realplay?DD452C8C-3AEF-4A7E-CDCC-AAF561F8328D"
    flv2 = "http://10.20.40.44:8080/realplay?640BFDCB-0AE2-1252-B879-7B52C6A4F569"
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())
    while cap.isOpened():
        ret, img_ori = cap.read()
        if ret:
            if queue.full():
                queue.get()
            else:
                queue.put(img_ori)
        else:
            cap.release()
            # read again
            cap = cv2.VideoCapture(flv1)


def process_data(_queue):
    while True:
        frame = _queue.get(True)
        cv2.namedWindow("YOLO3_Deep_SORT", 0)
        cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO3_Deep_SORT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # rtsp = "rtsp://admin:admin@10.10.0.253:80"
    # flv1 = "http://10.20.40.44:8080/realplay?DD452C8C-3AEF-4A7E-CDCC-AAF561F8328D"
    # flv2 = "http://10.20.40.44:8080/realplay?640BFDCB-0AE2-1252-B879-7B52C6A4F569"
    # cap = cv2.VideoCapture(flv1)

    # tradition_method(cap)

    q = Queue(maxsize=2)
    print("id of q is {}".format(id(q)))
    receive_data_no = Process(target=receive_data, args=(q,))
    receive_data_no.start()
    process_data(q)


