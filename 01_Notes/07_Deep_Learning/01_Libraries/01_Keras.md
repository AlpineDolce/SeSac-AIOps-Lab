<h2>Keras 핵심 개념 정리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Keras를 활용한 딥러닝 모델 개발의 전반적인 과정을 실무 관점에서 체계적으로 다룹니다. Keras의 핵심 API를 이용한 모델 구축, 데이터 전처리, 학습 및 평가, 그리고 모델 배포 전략까지, 실제 프로젝트에 바로 적용할 수 있는 실용적인 지식과 팁을 제공하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. Keras 개요 및 기본 개념](#1-keras-개요-및-기본-개념)
  - [1.1. Keras란? (TensorFlow와의 관계)](#11-keras란-tensorflow와의-관계)
  - [1.2. Keras의 철학 (User Friendliness, Modularity, Easy Extensibility)](#12-keras의-철학-user-friendliness-modularity-easy-extensibility)
  - [1.3. 설치 및 환경 설정](#13-설치-및-환경-설정)
  - [1.4. Keras의 주요 구성 요소 (Models, Layers, Optimizers, Losses, Metrics)](#14-keras의-주요-구성-요소-models-layers-optimizers-losses-metrics)

- [2. Keras 모델 구축](#2-keras-모델-구축)
  - [2.1. Sequential API를 이용한 모델 구축](#21-sequential-api를-이용한-모델-구축)
  - [2.2. Functional API를 이용한 복잡한 모델 구축](#22-functional-api를-이용한-복잡한-모델-구축)
  - [2.3. Subclassing API를 이용한 Custom Model 구현](#23-subclassing-api를-이용한-custom-model-구현)
  - [2.4. 사전 학습된 모델 (Pre-trained Models) 활용 (Application API)](#24-사전-학습된-모델-pre-trained-models-활용-application-api)

- [3. 데이터 전처리 및 로딩](#3-데이터-전처리-및-로딩)
  - [3.1. NumPy 배열을 이용한 데이터 준비](#31-numpy-배열을-이용한-데이터-준비)
  - [3.2. `tf.data`와 Keras 통합 (성능 최적화)](#32-tfdata와-keras-통합-성능-최적화)
  - [3.3. 이미지 데이터 전처리 및 증강 (`ImageDataGenerator`)](#33-이미지-데이터-전처리-및-증강-imagedatagenerator)
  - [3.4. 텍스트 데이터 전처리 (`Tokenizer`)](#34-텍스트-데이터-전처리-tokenizer)

- [4. 모델 학습, 평가 및 예측](#4-모델-학습-평가-및-예측)
  - [4.1. 모델 컴파일 (Optimizer, Loss, Metrics 설정)](#41-모델-컴파일-optimizer-loss-metrics-설정)
  - [4.2. `model.fit()`을 이용한 모델 학습](#42-modelfit을-이용한-모델-학습)
  - [4.3. `model.evaluate()`를 이용한 모델 평가](#43-modelevaluate를-이용한-모델-평가)
  - [4.4. `model.predict()`를 이용한 예측](#44-modelpredict를-이용한-예측)
  - [4.5. 콜백 (Callbacks) 활용 (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)](#45-콜백-callbacks-활용-earlystopping-modelcheckpoint-reducelronplateau)

- [5. 모델 저장 및 로드](#5-모델-저장-및-로드)
  - [5.1. 전체 모델 저장 및 로드 (SavedModel, H5)](#51-전체-모델-저장-및-로드-savedmodel-h5)
  - [5.2. 가중치만 저장 및 로드](#52-가중치만-저장-및-로드)
  - [5.3. 모델 아키텍처만 저장 및 로드](#53-모델-아키텍처만-저장-및-로드)

- [6. Keras 모델 배포 전략](#6-keras-모델-배포-전략)
  - [6.1. TensorFlow Serving을 이용한 배포](#61-tensorflow-serving을-이용한-배포)
  - [6.2. TensorFlow Lite를 이용한 모바일/엣지 디바이스 배포](#62-tensorflow-lite를-이용한-모바일엣지-디바이스-배포)
  - [6.3. TensorFlow.js를 이용한 웹 브라우저 배포](#63-tensorflowjs-웹-브라우저-배포)
  - [6.4. ONNX/TensorRT 변환을 통한 최적화](#64-onnx-tensorrt-변환을-통한-최적화)

- [7. TensorBoard를 활용한 시각화 및 디버깅](#7-tensorboard를-활용한-시각화-및-디버깅)
  - [7.1. TensorBoard 개요 및 설치](#71-tensorboard-개요-및-설치)
  - [7.2. 학습 과정 모니터링 (Scalars, Graphs, Histograms)](#72-학습-과정-모니터링-scalars-graphs-histograms)
  - [7.3. 모델 그래프 시각화](#73-모델-그래프-시각화)
  - [7.4. 임베딩 시각화](#74-임베딩-시각화)

- [8. 실전 프로젝트 예제 및 팁](#8-실전-프로젝트-예제-및-팁)
  - [8.1. 이미지 분류 (CNN)](#81-이미지-분류-cnn)
  - [8.2. 텍스트 분류 (RNN/Transformer)](#82-텍스트-분류-rnntransformer)
  - [8.3. 시계열 예측 (LSTM)](#83-시계열-예측-lstm)
  - [8.4. GAN (Generative Adversarial Networks) 기본](#84-gan-generative-adversarial-networks-기본)
  - [8.5. 모델 최적화 및 성능 튜닝 팁](#85-모델-최적화-및-성능-튜닝-팁)
  - [8.6. 에러 처리 및 디버깅 전략](#86-에러-처리-및-디버깅-전략)