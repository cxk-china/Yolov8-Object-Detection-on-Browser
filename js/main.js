// Add labels
const labels = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

// 中文标签
const chineseLabels = [
  "人",
  "自行车",
  "汽车",
  "摩托车",
  "飞机",
  "公交车",
  "火车",
  "卡车",
  "船",
  "交通灯",
  "消防栓",
  "停止标志",
  "停车计时器",
  "长凳",
  "鸟",
  "猫",
  "狗",
  "马",
  "羊",
  "牛",
  "大象",
  "熊",
  "斑马",
  "长颈鹿",
  "背包",
  "雨伞",
  "手提包",
  "领带",
  "行李箱",
  "飞盘",
  "滑雪板",
  "滑雪",
  "运动球",
  "风筝",
  "棒球棒",
  "棒球手套",
  "滑板",
  "冲浪板",
  "网球拍",
  "瓶子",
  "酒杯",
  "杯子",
  "叉子",
  "刀",
  "勺子",
  "碗",
  "香蕉",
  "苹果",
  "三明治",
  "橙子",
  "西兰花",
  "胡萝卜",
  "热狗",
  "披萨",
  "甜甜圈",
  "蛋糕",
  "椅子",
  "沙发",
  "盆栽植物",
  "床",
  "餐桌",
  "厕所",
  "电视",
  "笔记本电脑",
  "鼠标",
  "遥控器",
  "键盘",
  "手机",
  "微波炉",
  "烤箱",
  "烤面包机",
  "水槽",
  "冰箱",
  "书",
  "时钟",
  "花瓶",
  "剪刀",
  "泰迪熊",
  "吹风机",
  "牙刷",
];

// React State implementation in Vanilla JS
const useState = (defaultValue) => {
  let value = defaultValue;
  const getValue = () => value;
  const setValue = (newValue) => (value = newValue);
  return [getValue, setValue];
};

// Declare variables
const numClass = labels.length;
const [session, setSession] = useState(null);
let mySession;
// 跟踪当前活动模式
let activeMode = 'camera';

// 获取当前标签
const getCurrentLabels = () => {
  return chineseLabels;
};

// Declare DOM elements
const video = document.querySelector("#video");
const cameraCanvas = document.querySelector("#cameraCanvas");
const resultCanvas = document.querySelector("#resultCanvas");
const videoCanvas = document.querySelector("#videoCanvas");
const imagePreview = document.querySelector("#imagePreview");
const videoPlayer = document.querySelector("#videoPlayer");
const cameraModeDiv = document.querySelector("#cameraMode");
const imageModeDiv = document.querySelector("#imageMode");
const videoModeDiv = document.querySelector("#videoMode");
const zoomModeDiv = document.querySelector("#zoomMode");
const zoomContent = document.querySelector("#zoomContent");
let videoInterval = null;
let currentZoomScale = 1;
let currentZoomElement = null;
let translateX = 0;
let translateY = 0;

// Configs
const modelName = "yolov8n.onnx";
const modelInputShape = [1, 3, 416, 416];
const topk = 100;
const iouThreshold = 0.45;
const scoreThreshold = 0.2;

// wait until opencv.js initialized
cv["onRuntimeInitialized"] = async () => {
  // create session
  const [yolov8, nms] = await Promise.all([
    ort.InferenceSession.create(`model/${modelName}`),
    ort.InferenceSession.create(`model/nms-yolov8.onnx`),
  ]);
  // warmup main model
  const tensor = new ort.Tensor(
    "float32",
    new Float32Array(modelInputShape.reduce((a, b) => a * b)),
    modelInputShape
  );
  await yolov8.run({ images: tensor });

  mySession = setSession({ net: yolov8, nms: nms });
  
};

// Detect Image Function
const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  const [modelWidth, modelHeight] = inputShape.slice(2);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new ort.Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new ort.Tensor("float32", new Float32Array([topk, iouThreshold, scoreThreshold])); // nms config tensor
  const { output0 } = await session.net.run({ images: tensor }); // run session and get output layer
  const { selected } = await session.nms.run({ detection: output0, config: config }); // perform nms and filter boxes

  const boxes = [];

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
    const box = data.slice(0, 4);
    const scores = data.slice(4); // classes probability scores
    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores

    const [x, y, w, h] = [
      (box[0] - 0.5 * box[2]) * xRatio, // upscale left
      (box[1] - 0.5 * box[3]) * yRatio, // upscale top
      box[2] * xRatio, // upscale width
      box[3] * yRatio, // upscale height
    ]; // keep boxes in maxSize range

    boxes.push({
      label: label,
      probability: score,
      bounding: [x, y, w, h], // upscale box
    }); // update boxes to draw later
  }

  // 存储边界框数据，以便在语言切换时重新渲染
  canvas.boxesData = boxes;
  
  renderBoxes(canvas, boxes); // Draw boxes
  input.delete(); // delete unused Mat
};

// Render box
const renderBoxes = (canvas, boxes) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  // 如果是图片模式的结果画布，先绘制图片
  if (canvas === resultCanvas && imagePreview.complete) {
    // 绘制图片以填充整个画布
    ctx.drawImage(imagePreview, 0, 0, canvas.width, canvas.height);
  }

  const colors = new Colors();

  // 检查是否是图片模式或视频模式
  const isImageMode = canvas === resultCanvas;
  const isVideoMode = canvas === videoCanvas;

  // font configs
  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";

  boxes.forEach((box) => {
    const klass = getCurrentLabels()[box.label];
    const color = colors.get(box.label);
    const score = (box.probability * 100).toFixed(1);
    let [x1, y1, width, height] = box.bounding;
    
    // 如果是图片模式或视频模式，调整边界框坐标以匹配显示的图片/视频
    if (isImageMode || isVideoMode) {
      // 将检测坐标（基于416x416）映射到实际画布尺寸上
      const scaleX = canvas.width / 416;
      const scaleY = canvas.height / 416;
      
      x1 = x1 * scaleX;
      y1 = y1 * scaleY;
      width = width * scaleX;
      height = height * scaleY;
    }

    // draw box.
    ctx.fillStyle = Colors.hexToRgba(color, 0.2);
    ctx.fillRect(x1, y1, width, height);
    // draw border box
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(
      Math.min(ctx.canvas.width, ctx.canvas.height) / 200,
      2.5
    );
    ctx.strokeRect(x1, y1, width, height);

    // draw the label background.
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(klass + " - " + score + "%").width;
    const textHeight = parseInt(font, 10); // base 10
    const yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    );

    // Draw labels
    ctx.fillStyle = "#ffffff";
    ctx.fillText(
      klass + " - " + score + "%",
      x1 - 1,
      yText < 0 ? 1 : yText + 1
    );
  });
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight) => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};

class Colors {
  // ultralytics color palette https://ultralytics.com/
  constructor() {
    this.palette = [
      "#FF3838",
      "#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
      "#CB38FF",
      "#FF95C8",
      "#FF37C7",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];
  static hexToRgba = (hex, alpha) => {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? `rgba(${[
          parseInt(result[1], 16),
          parseInt(result[2], 16),
          parseInt(result[3], 16),
        ].join(", ")}, ${alpha})`
      : null;
  };
}

// 显示模式切换
const showMode = (mode) => {
  // 隐藏所有模式
  cameraModeDiv.style.display = 'none';
  imageModeDiv.style.display = 'none';
  videoModeDiv.style.display = 'none';
  zoomModeDiv.style.display = 'none';
  
  // 显示指定模式
  if (mode === 'camera') {
    cameraModeDiv.style.display = 'block';
    if (mode !== 'zoom') activeMode = 'camera';
  } else if (mode === 'image') {
    imageModeDiv.style.display = 'block';
    if (mode !== 'zoom') activeMode = 'image';
  } else if (mode === 'video') {
    videoModeDiv.style.display = 'block';
    if (mode !== 'zoom') activeMode = 'video';
  } else if (mode === 'zoom') {
    zoomModeDiv.style.display = 'block';
  }
};

// 放大查看功能
const showZoom = (element) => {
  // 重置缩放比例
  currentZoomScale = 1;
  currentZoomElement = element;
  
  // 清空之前的内容
  zoomContent.innerHTML = '';
  
  // 创建新元素用于放大查看
  let zoomElement;
  if (element.tagName === 'CANVAS') {
    zoomElement = document.createElement('canvas');
    zoomElement.width = element.width;
    zoomElement.height = element.height;
    const ctx = zoomElement.getContext('2d');
    ctx.drawImage(element, 0, 0, element.width, element.height);
  } else {
    zoomElement = document.createElement('img');
    zoomElement.src = element.src;
  }
  
  // 添加样式以确保居中显示
  zoomElement.id = 'zoomedElement';
  zoomElement.style.maxWidth = '90%';
  zoomElement.style.maxHeight = '90%';
  zoomElement.style.objectFit = 'contain';
  zoomElement.style.margin = 'auto';
  zoomElement.style.display = 'block';
  
  // 添加到DOM
  zoomContent.appendChild(zoomElement);
  
  // 显示放大模式
  showMode('zoom');
  
  // 添加鼠标滚轮缩放功能
  zoomContent.addEventListener('wheel', handleZoomWheel);
  
  // 添加拖动功能
  enableDragZoom(zoomElement);
};

// 处理鼠标滚轮缩放
const handleZoomWheel = (e) => {
  e.preventDefault();
  
  // 向下滚动缩小，向上滚动放大
  if (e.deltaY > 0) {
    // 缩小
    currentZoomScale -= 0.1;
    if (currentZoomScale < 0.1) currentZoomScale = 0.1;
  } else {
    // 放大
    currentZoomScale += 0.1;
    if (currentZoomScale > 5) currentZoomScale = 5;
  }
  
  applyZoom();
};

// 启用拖动功能
const enableDragZoom = (element) => {
  let isDragging = false;
  let startX, startY;
  
  // 鼠标按下开始拖动
  zoomContent.addEventListener('mousedown', (e) => {
    if (currentZoomScale > 1) {
      isDragging = true;
      startX = e.clientX - translateX;
      startY = e.clientY - translateY;
      zoomContent.style.cursor = 'grabbing';
    }
  });
  
  // 鼠标移动时拖动
  zoomContent.addEventListener('mousemove', (e) => {
    if (isDragging && currentZoomScale > 1) {
      translateX = e.clientX - startX;
      translateY = e.clientY - startY;
      
      // 限制拖动范围，防止图片被拖出视图
      const zoomedElement = document.getElementById('zoomedElement');
      const zoomedRect = zoomedElement.getBoundingClientRect();
      const containerRect = zoomContent.getBoundingClientRect();
      
      // 计算最大可拖动范围
      const maxTranslateX = (zoomedRect.width * currentZoomScale - containerRect.width) / 2;
      const maxTranslateY = (zoomedRect.height * currentZoomScale - containerRect.height) / 2;
      
      // 限制在范围内
      if (maxTranslateX > 0) {
        translateX = Math.max(-maxTranslateX, Math.min(translateX, maxTranslateX));
      } else {
        translateX = 0;
      }
      
      if (maxTranslateY > 0) {
        translateY = Math.max(-maxTranslateY, Math.min(translateY, maxTranslateY));
      } else {
        translateY = 0;
      }
      
      applyZoomWithTranslate();
    }
  });
  
  // 鼠标松开结束拖动
  window.addEventListener('mouseup', () => {
    isDragging = false;
    zoomContent.style.cursor = 'grab';
  });
  
  // 鼠标离开结束拖动
  zoomContent.addEventListener('mouseleave', () => {
    isDragging = false;
    zoomContent.style.cursor = 'grab';
  });
};

// 应用缩放和平移
const applyZoomWithTranslate = () => {
  const zoomedElement = document.getElementById('zoomedElement');
  if (!zoomedElement) return;
  
  zoomedElement.style.transform = `scale(${currentZoomScale}) translate(${translateX / currentZoomScale}px, ${translateY / currentZoomScale}px)`;
  zoomedElement.style.transformOrigin = 'center center';
};

// 应用缩放
const applyZoom = () => {
  const zoomedElement = document.getElementById('zoomedElement');
  if (!zoomedElement) return;
  
  // 重置平移
  translateX = 0;
  translateY = 0;
  
  zoomedElement.style.transform = `scale(${currentZoomScale})`;
  zoomedElement.style.transformOrigin = 'center center';
  
  // 更新鼠标样式
  if (currentZoomScale > 1) {
    zoomContent.style.cursor = 'grab';
  } else {
    zoomContent.style.cursor = 'default';
  }
};

// 重置UI状态
const resetUI = () => {
  // 停止视频处理
  if (videoInterval) {
    clearInterval(videoInterval);
    videoInterval = null;
  }
  
  // 停止摄像头
  if (video.srcObject) {
    let stream = video.srcObject;
    stream.getTracks().forEach(function (track) {    
        track.stop();    
    });
    video.srcObject = null;
  }
  
  // 停止视频播放
  videoPlayer.pause();
  videoPlayer.src = "";
  
  // 重置进度条
  if (progressBar) progressBar.style.width = "0%";
  if (playPauseBtn) playPauseBtn.textContent = "播放";
  
  // 显示默认模式（摄像头模式）
  showMode('camera');
  
  // 重置按钮状态
  document.querySelector("#runInference").style.display = "block";
  document.querySelector("#stopInference").style.display = "none";
  document.querySelector("#uploadImageBtn").style.display = "block";
  document.querySelector("#uploadVideoBtn").style.display = "block";
  document.querySelector("#processImage").style.display = "none";
  document.querySelector("#processVideo").style.display = "none";
  document.querySelector("#backToMain").style.display = "none";
  
  // 清除画布
  const clearCanvas = (canvas) => {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    // 清除存储的图片参数
    delete canvas.imageOffsetX;
    delete canvas.imageOffsetY;
    delete canvas.imageDrawWidth;
    delete canvas.imageDrawHeight;
    delete canvas.imageRatio;
  };
  
  clearCanvas(cameraCanvas);
  clearCanvas(resultCanvas);
  clearCanvas(videoCanvas);
};

// 摄像头检测
document.querySelector("#runInference").addEventListener("click", () => {
  resetUI();
  
  document.querySelector("#runInference").style.display = "none";
  document.querySelector("#uploadImageBtn").style.display = "none";
  document.querySelector("#uploadVideoBtn").style.display = "none";
  document.querySelector("#backToMain").style.display = "block";
  document.querySelector("#stopInference").style.display = "block";
  
  showMode('camera');
  
  // Set video stream constraints
  const constraints = {
    audio: false,
    video: { width: 640, height: 480, facingMode: "environment" },
  };

  // Request access to the user's camera
  navigator.mediaDevices
    .getUserMedia(constraints)
    .then((stream) => {
      video.srcObject = stream;
      video.play();
      videoInterval = setInterval(() => {
        // 获取视频尺寸
        const videoWidth = video.videoWidth || 640;
        const videoHeight = video.videoHeight || 480;
        const videoRatio = videoWidth / videoHeight;
        const canvasRatio = cameraCanvas.width / cameraCanvas.height;
        
        let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
        
        if (videoRatio > canvasRatio) {
          // 视频更宽
          drawWidth = cameraCanvas.width;
          drawHeight = cameraCanvas.width / videoRatio;
          offsetY = (cameraCanvas.height - drawHeight) / 2;
        } else {
          // 视频更高
          drawHeight = cameraCanvas.height;
          drawWidth = cameraCanvas.height * videoRatio;
          offsetX = (cameraCanvas.width - drawWidth) / 2;
        }
        
        // Draw video frame on canvas
        const context = cameraCanvas.getContext("2d");
        context.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
        context.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);
        
        // 创建一个临时canvas用于检测
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 416;
        tempCanvas.height = 416;
        const tempContext = tempCanvas.getContext('2d');
        tempContext.drawImage(video, 0, 0, 416, 416);
        
        // 存储绘图参数以供后续使用
        cameraCanvas.imageOffsetX = offsetX;
        cameraCanvas.imageOffsetY = offsetY;
        cameraCanvas.imageDrawWidth = drawWidth;
        cameraCanvas.imageDrawHeight = drawHeight;
        cameraCanvas.imageRatio = videoRatio;

        // Run object detection on the canvas image
        detectImage(
          tempCanvas,
          cameraCanvas,
          mySession,
          topk,
          iouThreshold,
          scoreThreshold,
          modelInputShape
        );
      }, 100);
    })
    .catch((err) => {
      console.error(err);
      alert("无法访问摄像头：" + err.message);
      resetUI();
    });
});

// 停止检测
document.querySelector("#stopInference").addEventListener("click", resetUI);

// 返回主界面
document.querySelector("#backToMain").addEventListener("click", resetUI);

// 图片上传按钮点击事件
document.querySelector("#uploadImageBtn").addEventListener("click", () => {
  document.querySelector("#imageUpload").click();
});

// 视频上传按钮点击事件
document.querySelector("#uploadVideoBtn").addEventListener("click", () => {
  document.querySelector("#videoUpload").click();
});

// 处理图片上传
document.querySelector("#imageUpload").addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    resetUI();
    
    const file = e.target.files[0];
    const reader = new FileReader();
    
    reader.onload = (event) => {
      imagePreview.src = event.target.result;
      imagePreview.onload = () => {
        showMode('image');
        document.querySelector("#runInference").style.display = "none";
        document.querySelector("#uploadImageBtn").style.display = "none";
        document.querySelector("#uploadVideoBtn").style.display = "none";
        document.querySelector("#processImage").style.display = "block";
        document.querySelector("#backToMain").style.display = "block";
      };
    };
    
    reader.readAsDataURL(file);
  }
});

// 处理视频上传
document.querySelector("#videoUpload").addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    resetUI();
    
    const file = e.target.files[0];
    
    videoPlayer.src = URL.createObjectURL(file);
    showMode('video');
    document.querySelector("#runInference").style.display = "none";
    document.querySelector("#uploadImageBtn").style.display = "none";
    document.querySelector("#uploadVideoBtn").style.display = "none";
    document.querySelector("#processVideo").style.display = "block";
    document.querySelector("#backToMain").style.display = "block";
  }
});

// 处理图片
document.querySelector("#processImage").addEventListener("click", () => {
  if (!imagePreview.complete) {
    alert("图片尚未加载完成，请稍候再试");
    return;
  }
  
  // 获取原图的实际显示尺寸
  const imgRect = imagePreview.getBoundingClientRect();
  const displayWidth = imgRect.width;
  const displayHeight = imgRect.height;
  
  // 调整resultCanvas的尺寸以匹配原图比例和显示大小
  const imageRatio = imagePreview.naturalWidth / imagePreview.naturalHeight;
  
  // 设置画布尺寸与原图显示尺寸相同
  resultCanvas.width = imagePreview.naturalWidth;
  resultCanvas.height = imagePreview.naturalHeight;
  
  // 创建一个临时canvas用于检测
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 416;
  tempCanvas.height = 416;
  const tempContext = tempCanvas.getContext('2d');
  tempContext.drawImage(imagePreview, 0, 0, 416, 416);
  
  // 运行对象检测
  detectImage(
    tempCanvas,
    resultCanvas,
    mySession,
    topk,
    iouThreshold,
    scoreThreshold,
    modelInputShape
  );
});

// 处理视频
document.querySelector("#processVideo").addEventListener("click", () => {
  if (videoInterval) {
    // 如果已经在处理，则停止
    clearInterval(videoInterval);
    videoInterval = null;
    document.querySelector("#processVideo").textContent = "处理视频";
    // 同时暂停视频
    videoPlayer.pause();
    return;
  }
  
  document.querySelector("#processVideo").textContent = "停止处理";
  
  // 确保视频已加载
  if (videoPlayer.readyState < 2) {
    videoPlayer.addEventListener('loadeddata', startVideoProcessing);
  } else {
    startVideoProcessing();
  }
});

function startVideoProcessing() {
  // 播放视频
  videoPlayer.play();
  
  // 更新播放/暂停按钮状态
  if (playPauseBtn) playPauseBtn.textContent = '暂停';
  
  // 处理视频帧的函数
  const processFrame = () => {
    // 调整canvas大小以匹配视频尺寸
    const videoWidth = videoPlayer.videoWidth;
    const videoHeight = videoPlayer.videoHeight;
    const videoRatio = videoWidth / videoHeight;
    
    // 获取视频元素的当前显示尺寸
    const videoRect = videoPlayer.getBoundingClientRect();
    const displayWidth = videoRect.width;
    const displayHeight = videoRect.height;
    
    // 设置canvas尺寸与视频显示尺寸相同
    videoCanvas.width = videoWidth;
    videoCanvas.height = videoHeight;
    videoCanvas.style.width = `${displayWidth}px`;
    videoCanvas.style.height = `${displayHeight}px`;
    
    const context = videoCanvas.getContext("2d");
    
    // 清除画布
    context.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
    
    // 创建一个临时canvas用于检测
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 416;
    tempCanvas.height = 416;
    const tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(videoPlayer, 0, 0, 416, 416);
    
    // 运行对象检测
    detectImage(
      tempCanvas,
      videoCanvas,
      mySession,
      topk,
      iouThreshold,
      scoreThreshold,
      modelInputShape
    );
  };
  
  // 初次处理
  processFrame();
  
  // 每隔一定时间对视频帧进行处理
  videoInterval = setInterval(() => {
    // 即使视频暂停，也能显示最后一帧的检测结果
    if (!videoPlayer.paused) {
      processFrame();
    }
  }, 100);
  
  // 视频结束时停止处理
  videoPlayer.addEventListener('ended', () => {
    if (videoInterval) {
      clearInterval(videoInterval);
      videoInterval = null;
      document.querySelector("#processVideo").textContent = "处理视频";
      if (playPauseBtn) playPauseBtn.textContent = '播放';
    }
  });
}

// 获取当前模式
const getCurrentMode = () => {
  return activeMode;
};

// 放大按钮事件
document.getElementById('zoomIn').addEventListener('click', () => {
  currentZoomScale += 0.2;
  if (currentZoomScale > 5) currentZoomScale = 5;
  applyZoom();
});

// 缩小按钮事件
document.getElementById('zoomOut').addEventListener('click', () => {
  currentZoomScale -= 0.2;
  if (currentZoomScale < 0.2) currentZoomScale = 0.2;
  applyZoom();
});

// 重置按钮事件
document.getElementById('zoomReset').addEventListener('click', () => {
  currentZoomScale = 1;
  applyZoom();
});

// 关闭放大查看
document.querySelector('.close-zoom').addEventListener('click', () => {
  currentZoomScale = 1;
  translateX = 0;
  translateY = 0;
  showMode(activeMode);
});

// 为图片和画布添加点击事件，启用放大查看
document.addEventListener('DOMContentLoaded', () => {
  // 为原始图片添加点击事件
  imagePreview.addEventListener('click', () => {
    showZoom(imagePreview);
  });
  
  // 为结果画布添加点击事件
  resultCanvas.addEventListener('click', () => {
    showZoom(resultCanvas);
  });
});

// 为视频播放器添加事件监听器，在视频尺寸变化时调整画布
videoPlayer.addEventListener('loadedmetadata', () => {
  if (activeMode === 'video') {
    // 获取视频元素的当前显示尺寸
    const videoRect = videoPlayer.getBoundingClientRect();
    const displayWidth = videoRect.width;
    const displayHeight = videoRect.height;
    
    // 设置canvas尺寸与视频显示尺寸相同
    videoCanvas.width = videoPlayer.videoWidth;
    videoCanvas.height = videoPlayer.videoHeight;
    videoCanvas.style.width = `${displayWidth}px`;
    videoCanvas.style.height = `${displayHeight}px`;
  }
});

// 为视频播放器添加事件监听器，处理视频点击放大功能
videoPlayer.addEventListener('click', (e) => {
  // 如果点击的是视频控件区域，不处理点击事件
  if (e.target !== videoPlayer || !videoInterval) return;
  
  // 阻止事件冒泡，防止触发其他点击事件
  e.stopPropagation();
  
  // 点击视频时暂停/播放
  if (videoPlayer.paused) {
    videoPlayer.play();
  } else {
    videoPlayer.pause();
  }
});

// 视频控制按钮
const playPauseBtn = document.getElementById('playPauseBtn');
const progressBar = document.getElementById('progressBar');
const progressContainer = document.querySelector('.progress-container');

// 播放/暂停按钮点击事件
playPauseBtn.addEventListener('click', () => {
  if (!videoInterval) return; // 只在处理视频时响应
  
  if (videoPlayer.paused) {
    videoPlayer.play();
    playPauseBtn.textContent = '暂停';
  } else {
    videoPlayer.pause();
    playPauseBtn.textContent = '播放';
  }
});

// 视频时间更新事件
videoPlayer.addEventListener('timeupdate', () => {
  if (!videoInterval) return;
  
  // 更新进度条
  const progress = (videoPlayer.currentTime / videoPlayer.duration) * 100;
  progressBar.style.width = `${progress}%`;
});

// 进度条点击事件
progressContainer.addEventListener('click', (e) => {
  if (!videoInterval) return;
  
  // 计算点击位置相对于进度条的比例
  const rect = progressContainer.getBoundingClientRect();
  const pos = (e.clientX - rect.left) / rect.width;
  
  // 设置视频当前时间
  videoPlayer.currentTime = pos * videoPlayer.duration;
});

// 视频播放状态变化事件
videoPlayer.addEventListener('play', () => {
  if (playPauseBtn) playPauseBtn.textContent = '暂停';
});

videoPlayer.addEventListener('pause', () => {
  if (playPauseBtn) playPauseBtn.textContent = '播放';
});