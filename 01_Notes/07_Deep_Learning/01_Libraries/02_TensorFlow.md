<h2>TensorFlow 핵심 개념 정리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 TensorFlow를 활용한 딥러닝 모델 개발 및 배포에 필요한 핵심 개념과 실무 기술을 체계적으로 정리합니다. 데이터 전처리부터 모델 구축, 학습, 평가, 그리고 실제 서비스 환경에서의 배포까지 전 과정을 다루어, 독자가 TensorFlow를 실제 프로젝트에 효과적으로 적용할 수 있도록 돕는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. TensorFlow 개요 및 기본 개념](#1-tensorflow-개요-및-기본-개념)
  - [1.1. TensorFlow란?](#11-tensorflow란)
  - [1.2. TensorFlow 2.x의 특징 (Eager Execution, Keras API)](#12-tensorflow-2x의-특징-eager-execution-keras-api)
  - [1.3. 설치 및 환경 설정 (Python, GPU 설정)](#13-설치-및-환경-설정-python-gpu-설정)
  - [1.4. 기본 데이터 구조: Tensor](#14-기본-데이터-구조-tensor)
  - [1.5. 자동 미분 (Automatic Differentiation)과 GradientTape](#15-자동-미분-automatic-differentiation과-gradienttape)

- [2. Keras API를 활용한 모델 구축](#2-keras-api를-활용한-모델-구축)
  - [2.1. Sequential API](#21-sequential-api)
  - [2.2. Functional API](#22-functional-api)
  - [2.3. Custom Layer 및 Custom Model 구현](#23-custom-layer-및-custom-model-구현)
  - [2.4. 모델 컴파일, 학습 (fit), 평가 (evaluate), 예측 (predict)](#24-모델-컴파일-학습-fit-평가-evaluate-예측-predict)
  - [2.5. 콜백 (Callbacks) 활용 (EarlyStopping, ModelCheckpoint)](#25-콜백-callbacks-활용-earlystopping-modelcheckpoint)

- [3. 데이터 파이프라인 구축 (tf.data)](#3-데이터-파이프라인-구축-tfdata)
  - [3.1. `tf.data.Dataset` 개요](#31-tfdataDataset-개요)
  - [3.2. 데이터 로드 (from_tensor_slices, from_generator)](#32-데이터-로드-from_tensor_slices-from_generator)
  - [3.3. 데이터 전처리 및 증강 (map, filter, batch, shuffle)](#33-데이터-전처리-및-증강-map-filter-batch-shuffle)
  - [3.4. 성능 최적화 (prefetch, cache)](#34-성능-최적화-prefetch-cache)

- [4. 모델 저장 및 로드](#4-모델-저장-및-로드)
  - [4.1. Keras H5 형식](#41-keras-h5-형식)
  - [4.2. SavedModel 형식](#42-savedmodel-형식)
  - [4.3. 체크포인트 (Checkpoints) 관리](#43-체크포인트-checkpoints-관리)

- [5. 고급 모델링 및 학습 기법](#5-고급-모델링-및-학습-기법)
  - [5.1. 전이 학습 (Transfer Learning)](#51-전이-학습-transfer-learning)
  - [5.2. Fine-tuning](#52-fine-tuning)
  - [5.3. 분산 학습 (Distributed Training)](#53-분산-학습-distributed-training)
  - [5.4. Mixed Precision Training](#54-mixed-precision-training)

- [6. TensorFlow Serving 및 배포](#6-tensorflow-serving-및-배포)
  - [6.1. TensorFlow Serving 개요](#61-tensorflow-serving-개요)
  - [6.2. Docker를 이용한 Serving 환경 구축](#62-docker를-이용한-serving-환경-구축)
  - [6.3. RESTful API를 통한 모델 추론](#63-restful-api를-통한-모델-추론)
  - [6.4. TensorFlow Lite (모바일 및 엣지 디바이스)](#64-tensorflow-lite-모바일-및-엣지-디바이스)
  - [6.5. TensorFlow.js (웹 브라우저 배포)](#65-tensorflowjs-웹-브라우저-배포)

- [7. TensorBoard를 활용한 시각화 및 디버깅](#7-tensorboard를-활용한-시각화-및-디버깅)
  - [7.1. TensorBoard 개요 및 설치](#71-tensorboard-개요-및-설치)
  - [7.2. 학습 과정 모니터링 (Scalars, Graphs, Histograms)](#72-학습-과정-모니터링-scalars-graphs-histograms)
  - [7.3. 모델 그래프 시각화](#73-모델-그래프-시각화)
  - [7.4. 임베딩 시각화](#74-임베딩-시각화)

- [8. 실전 프로젝트 예제 및 팁](#8-실전-프로젝트-예제-및-팁)
  - [8.1. 이미지 분류 (CNN)](#81-이미지-분류-cnn)
  - [8.2. 텍스트 분류 (RNN/Transformer)](#82-텍스트-분류-rnntransformer)
  - [8.3. 객체 탐지 (Object Detection)](#83-객체-탐지-object-detection)
  - [8.4. 모델 최적화 및 성능 튜닝 팁](#84-모델-최적화-및-성능-튜닝-팁)
  - [8.5. 에러 처리 및 디버깅 전략](#85-에러-처리-및-디버깅-전략)