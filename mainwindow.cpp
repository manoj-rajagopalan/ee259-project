﻿/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2018, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
copyright notice, this list of conditions and the
following disclaimer.

* Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the
following disclaimer in the documentation and/or other
materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
contributors may be used to endorse or promote products
derived from this software without specific prior
written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------
*/

#include <iostream>

#include "mainwindow.hpp"
#include "ui_mainwindow.h"

// Header files, Assimp.
#include <assimp/Exporter.hpp>
#include <assimp/postprocess.h>

#ifndef __unused
	#define __unused	__attribute__((unused))
#endif // __unused

using namespace Assimp;

inline
float3 subtract(float3 const& a, aiVector3D const& b)
{
	return make_float3(a.x - b[0], a.y - b[1], a.z - b[2]);
}

inline
float3 cross(float3 const& a, float3 const& b)
{
	float3 c;
	c.x = a.y*b.z - a.z*b.y;
	c.y = a.z*b.x - a.x*b.z;
	c.z = a.x*b.y - b.x*a.y;
	return c;
}

inline
float3 normalize(float3 const& v)
{
	float const v_len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
	if(v_len == 0.0f) {
		return v;
	}
	else {
		return make_float3(v.x/v_len, v.y/v_len, v.z/v_len);
	}
}

void MainWindow::ImportFile(const QString &pFileName) {
    QTime time_begin = QTime::currentTime();

	if ( mScene != nullptr ) {
		mImporter.FreeScene();
		mGLView->FreeScene();
	}

	// Try to import scene.
	mScene = mImporter.ReadFile(pFileName.toStdString(), aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_ValidateDataStructure | \
														 aiProcess_GenUVCoords | aiProcess_TransformUVCoords | aiProcess_FlipUVs);
	if ( mScene != nullptr ) {
		ui->lblLoadTime->setText(QString::number(time_begin.secsTo(QTime::currentTime())));
		LogInfo("Scene has " + QString::number(mScene->mNumMeshes) + " meshes");
		LogInfo("Import done: " + pFileName);
		// Prepare widgets for new scene.
		ui->leFileName->setText(pFileName.right(pFileName.length() - pFileName.lastIndexOf('/') - 1));
		ui->lstLight->clear();
		ui->lstCamera->clear();
		ui->cbxLighting->setChecked(true);	mGLView->Lighting_Enable();
		ui->cbxBBox->setChecked(false);		mGLView->Enable_SceneBBox(false);
		ui->cbxTextures->setChecked(true);	mGLView->Enable_Textures(true);

		//
		// Fill info labels
		//
		// Cameras
		ui->lblCameraCount->setText(QString::number(mScene->mNumCameras));
		// Lights
		ui->lblLightCount->setText(QString::number(mScene->mNumLights));
		// Meshes, faces, vertices.
		size_t qty_face = 0;
		size_t qty_vert = 0;

		for(size_t idx_mesh = 0; idx_mesh < mScene->mNumMeshes; idx_mesh++) {
			qty_face += mScene->mMeshes[idx_mesh]->mNumFaces;
			qty_vert += mScene->mMeshes[idx_mesh]->mNumVertices;
		}

		ui->lblMeshCount->setText(QString::number(mScene->mNumMeshes));
		ui->lblFaceCount->setText(QString::number(qty_face));
		ui->lblVertexCount->setText(QString::number(qty_vert));
		// Animation
		if(mScene->mNumAnimations)
			ui->lblHasAnimation->setText("yes");
		else
			ui->lblHasAnimation->setText("no");

		//
		// Set scene for GL viewer.
		//
		mGLView->SetScene(mScene, pFileName);
		// Select first camera
		ui->lstCamera->setCurrentRow(0);
		mGLView->Camera_Set(0);
		// Scene is loaded, do first rendering.
		LogInfo("Scene is ready for rendering.");
#if ASSIMP_QT4_VIEWER
		mGLView->updateGL();
#else
		mGLView->update();
#endif // ASSIMP_QT4_VIEWER

		// -- manojr -- Perform ray-tracing and visualize the point-cloud

		// Initialize transmitter
		aiMatrix4x4 cameraToWorld; // rows represent camera axes in world coords
		aiVector3D cameraPosition;
		mGLView->Camera_Matrix(/*transpose of*/ cameraToWorld, mSceneToWorldRotation, cameraPosition);
		cameraToWorld.Transpose();
		// Co-locate transmitter with camera
		transmitter_.position.x = cameraPosition[0];
		transmitter_.position.y = cameraPosition[1];
		transmitter_.position.z = cameraPosition[2];

		// UpdateTransmitterPose_(transmitter_, cameraToWorld);
		aiVector3D const& sceneCenter = mGLView->SceneCenter();
		transmitter_.zUnitVector = normalize(subtract(transmitter_.position, sceneCenter));
		transmitter_.xUnitVector = normalize(cross(make_float3(0,1,0), transmitter_.zUnitVector));
		transmitter_.yUnitVector = normalize(cross(transmitter_.zUnitVector,
		                                           transmitter_.xUnitVector));

		transmitter_.width = 1.0f;
		transmitter_.height = 1.0f;
		transmitter_.focalLength = 1.0f;
		transmitter_.numRays_x = 30;
		transmitter_.numRays_y = 30;
	
		std::unique_lock<std::mutex> rayTracingCommandLock(mRayTracingCommandMutex);		
		mAssimpOptixRayTracer.setScene(mScene);
        mAssimpOptixRayTracer.setSceneTransform(&mSceneToWorldRotation);
		mAssimpOptixRayTracer.setTransmitter(&transmitter_);
		mAssimpOptixRayTracer.setTransmitterTransform();
		rayTracingCommandLock.unlock();
		mRayTracingCommandConditionVariable.notify_one();
	}
	else
	{
		ResetSceneInfos();

		QString errorMessage = QString("Error parsing \'%1\' : \'%2\'").arg(pFileName).arg(mImporter.GetErrorString());
		QMessageBox::critical(this, "Import error", errorMessage);
		LogError(errorMessage);
	}// if(mScene != nullptr)
}

void MainWindow::UpdateTransmitterPose_(manojr::Transmitter& transmitter,
                                        aiMatrix4x4 const& cameraToWorld)
{
	transmitter.xUnitVector.x = cameraToWorld[0][0];
	transmitter.xUnitVector.y = cameraToWorld[1][0];
	transmitter.xUnitVector.z = cameraToWorld[2][0];
	
	transmitter.yUnitVector.x = cameraToWorld[0][1];
	transmitter.yUnitVector.y = cameraToWorld[1][1];
	transmitter.yUnitVector.z = cameraToWorld[2][1];
	
	transmitter.zUnitVector.x = cameraToWorld[0][2];
	transmitter.zUnitVector.y = cameraToWorld[1][2];
	transmitter.zUnitVector.z = cameraToWorld[2][2];
}

void MainWindow::ResetSceneInfos()
{
	ui->lblLoadTime->clear();
	ui->leFileName->clear();
	ui->lblMeshCount->setText("0");
	ui->lblFaceCount->setText("0");
	ui->lblVertexCount->setText("0");
	ui->lblCameraCount->setText("0");
	ui->lblLightCount->setText("0");
	ui->lblHasAnimation->setText("no");
}

/********************************************************************/
/************************ Logging functions *************************/
/********************************************************************/

void MainWindow::LogInfo(const QString& pMessage)
{
	Assimp::DefaultLogger::get()->info(pMessage.toStdString());
}

void MainWindow::LogError(const QString& pMessage)
{
	Assimp::DefaultLogger::get()->error(pMessage.toStdString());
}

/********************************************************************/
/*********************** Override functions ************************/
/********************************************************************/

void MainWindow::mousePressEvent(QMouseEvent* pEvent)
{
    const QPoint ms_pt = pEvent->pos();
    aiVector3D temp_v3;

	// Check if GLView is pointed.
	if(childAt(ms_pt) == mGLView)
	{
		if(!mMouse_Transformation.Position_Pressed_Valid)
		{
			mMouse_Transformation.Position_Pressed_Valid = true;// set flag
			// Store current transformation matrices.
			mGLView->Camera_Matrix(mMouse_Transformation.Rotation_AroundCamera, mMouse_Transformation.Rotation_Scene, temp_v3);
			mMouse_Transformation.Scene_Rotated = false;
			mMouse_Transformation.Camera_Rotated = false;
		}

		if(pEvent->button() & Qt::LeftButton)
			mMouse_Transformation.Position_Pressed_LMB = ms_pt;
		else if(pEvent->button() & Qt::RightButton)
			mMouse_Transformation.Position_Pressed_RMB = ms_pt;
	}
	else
	{
		mMouse_Transformation.Position_Pressed_Valid = false;
	}
}

/// @brief Perform ray-tracing and display results
/// @param pEvent 
void MainWindow::mouseReleaseEvent(QMouseEvent *pEvent)
{
	if(pEvent->buttons() == 0) {
		mMouse_Transformation.Position_Pressed_Valid = false;
		aiMatrix4x4 cameraToWorldRotation;
		aiVector3D cameraTranslation;
		mGLView->Camera_Matrix(/*transpose of*/ cameraToWorldRotation, mSceneToWorldRotation, cameraTranslation);
		cameraToWorldRotation.Transpose(); // all good now

		// Trigger ray-tracing.
		std::unique_lock<std::mutex> rayTracingCommandMutexLock(mRayTracingCommandMutex);
		if(mMouse_Transformation.Scene_Rotated) {
			mAssimpOptixRayTracer.setSceneTransform(&mSceneToWorldRotation);
			mMouse_Transformation.Scene_Rotated = false;
		}
		if(mMouse_Transformation.Camera_Rotated) {
			UpdateTransmitterPose_(transmitter_, cameraToWorldRotation);
			mAssimpOptixRayTracer.setTransmitterTransform();
			mMouse_Transformation.Camera_Rotated = false;
		}
		rayTracingCommandMutexLock.unlock();
		mRayTracingCommandConditionVariable.notify_one();
	}

}

void MainWindow::renderRayTracedPointCloud()
{
	std::unique_lock<std::mutex> lock(mRayTracingResultMutex);
	aiScene const *const rayTracedPointCloud = mAssimpOptixRayTracer.rayTracingResult(); // Ownership transfer
	mRayTracingResult.reset(rayTracedPointCloud); // Ownership transfer
	mGLView_rayTraced->SetScene(rayTracedPointCloud, QString());
	lock.unlock();
#if ASSIMP_QT4_VIEWER
	mGLView_rayTraced->updateGL();
#else
	mGLView_rayTraced->update();
#endif // ASSIMP_QT4_VIEWER
}

void MainWindow::mouseMoveEvent(QMouseEvent* pEvent)
{
	if(mMouse_Transformation.Position_Pressed_Valid)
	{
		if(pEvent->buttons() & Qt::LeftButton)
		{
			GLfloat dx = 180 * GLfloat(pEvent->x() - mMouse_Transformation.Position_Pressed_LMB.x()) / mGLView->width();
			GLfloat dy = 180 * GLfloat(pEvent->y() - mMouse_Transformation.Position_Pressed_LMB.y()) / mGLView->height();

			if(pEvent->modifiers() & Qt::ShiftModifier)
				mGLView->Camera_RotateScene(dy, 0, dx, &mMouse_Transformation.Rotation_Scene);// Rotate around oX and oZ axises.
			else
				mGLView->Camera_RotateScene(dy, dx, 0, &mMouse_Transformation.Rotation_Scene);// Rotate around oX and oY axises.
			
			mMouse_Transformation.Camera_Rotated = true;

#if ASSIMP_QT4_VIEWER
			mGLView->updateGL();
#else
			mGLView->update();
#endif // ASSIMP_QT4_VIEWER
		}

		if(pEvent->buttons() & Qt::RightButton)
		{
			GLfloat dx = 180 * GLfloat(pEvent->x() - mMouse_Transformation.Position_Pressed_RMB.x()) / mGLView->width();
			GLfloat dy = 180 * GLfloat(pEvent->y() - mMouse_Transformation.Position_Pressed_RMB.y()) / mGLView->height();

			if(pEvent->modifiers() & Qt::ShiftModifier)
				mGLView->Camera_Rotate(dy, 0, dx, &mMouse_Transformation.Rotation_AroundCamera);// Rotate around oX and oZ axises.
			else
				mGLView->Camera_Rotate(dy, dx, 0, &mMouse_Transformation.Rotation_AroundCamera);// Rotate around oX and oY axises.

			mMouse_Transformation.Scene_Rotated = true;

#if ASSIMP_QT4_VIEWER
			mGLView->updateGL();
#else
			mGLView->update();
#endif // ASSIMP_QT4_VIEWER
		}
	}
}

void MainWindow::keyPressEvent(QKeyEvent* pEvent)
{
	GLfloat step;

	if(pEvent->modifiers() & Qt::ControlModifier)
		step = 10;
	else if(pEvent->modifiers() & Qt::AltModifier)
		step = 100;
	else
		step = 1;

	if(pEvent->key() == Qt::Key_A)
		mGLView->Camera_Translate(-step, 0, 0);
	else if(pEvent->key() == Qt::Key_D)
		mGLView->Camera_Translate(step, 0, 0);
	else if(pEvent->key() == Qt::Key_W)
		mGLView->Camera_Translate(0, step, 0);
	else if(pEvent->key() == Qt::Key_S)
		mGLView->Camera_Translate(0, -step, 0);
	else if(pEvent->key() == Qt::Key_Up)
		mGLView->Camera_Translate(0, 0, -step);
	else if(pEvent->key() == Qt::Key_Down)
		mGLView->Camera_Translate(0, 0, step);

#if ASSIMP_QT4_VIEWER
	mGLView->updateGL();
#else
	mGLView->update();
#endif // ASSIMP_QT4_VIEWER
}

/********************************************************************/
/********************** Constructor/Destructor **********************/
/********************************************************************/

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent),
	  ui(new Ui::MainWindow),
	  mScene(nullptr),
	  mAssimpOptixRayTracer(mRayTracingCommandMutex,
	                        mRayTracingCommandConditionVariable,
							mRayTracingResultMutex)
{

	// other variables
	mMouse_Transformation.Position_Pressed_Valid = false;

	ui->setupUi(this);
	QGridLayout *const gridLayout = new QGridLayout(this);

	// Create OpenGL widget
	mGLView = new CGLView(this);
	mGLView->setMinimumSize(400, 300);
	mGLView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	mGLView->setFocusPolicy(Qt::StrongFocus);
	// Connect to GLView signals.
	connect(mGLView, SIGNAL(Paint_Finished(size_t, GLfloat)), SLOT(Paint_Finished(size_t, GLfloat)));
	connect(mGLView, SIGNAL(SceneObject_Camera(QString)), SLOT(SceneObject_Camera(QString)));
	connect(mGLView, SIGNAL(SceneObject_LightSource(QString)), SLOT(SceneObject_LightSource(QString)));
	gridLayout->addWidget(mGLView, 0, 0);

	// Create OpenGL widget for OptiX
	mGLView_rayTraced = new manojr::CRayTracingGLView(this, mRayTracingResultMutex);
	mGLView_rayTraced->setMinimumSize(400, 300);
	mGLView_rayTraced->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	mGLView_rayTraced->setFocusPolicy(Qt::StrongFocus);
	gridLayout->addWidget(mGLView_rayTraced, 0, 1);

	// and add it to layout
	ui->hlMainView->addLayout(gridLayout);
	// Create logger
	mLoggerView = new CLoggerView(ui->tbLog);
	DefaultLogger::create("", Logger::VERBOSE);
	DefaultLogger::get()->attachStream(mLoggerView, DefaultLogger::Debugging | DefaultLogger::Info | DefaultLogger::Err | DefaultLogger::Warn);

	ResetSceneInfos();

	// std::function<void(std::string const&)> logInfo = [this](std::string const& s) { LogInfo(s.c_str()); } ;
	// std::function<void(std::string const&)> logError = [this](std::string const& s) { LogError(s.c_str()); } ;
	// mAssimpOptixRayTracer.registerLoggingFunctions(logInfo, logError);
	std::function<void(std::string const&)> infoToCout = [](std::string const& s) { std::cout << s << std::endl; };
	std::function<void(std::string const&)> errToCerr = [](std::string const& s) { std::cerr << s << std::endl; };
	mAssimpOptixRayTracer.registerLoggingFunctions(infoToCout, errToCerr);
	connect(&mAssimpOptixRayTracer, SIGNAL(rayTracingComplete()), SLOT(renderRayTracedPointCloud()));
	mRayTracingThread = std::thread([&]() { mAssimpOptixRayTracer.eventLoop(); });	
}

MainWindow::~MainWindow()
{
	{ // Signal ray-tracing thread to quit
		LogInfo(tr("Signaled ray-tracing thread to quit."));
		std::unique_lock<std::mutex> lock(mRayTracingCommandMutex);
		mAssimpOptixRayTracer.quit();
		mRayTracingCommandConditionVariable.notify_one();
	}
	mRayTracingThread.join();
	LogInfo(tr("Ray-tracing thread has joined."));

	using namespace Assimp;

	DefaultLogger::get()->detachStream(mLoggerView, DefaultLogger::Debugging | DefaultLogger::Info | DefaultLogger::Err | DefaultLogger::Warn);
	DefaultLogger::kill();

	if(mScene != nullptr) mImporter.FreeScene();
	if(mLoggerView != nullptr) delete mLoggerView;
	if(mGLView != nullptr) delete mGLView;
	delete ui;
}

/********************************************************************/
/****************************** Slots *******************************/
/********************************************************************/

void MainWindow::Paint_Finished(const size_t pPaintTime_ms, const GLfloat pDistance)
{
	ui->lblRenderTime->setText(QString::number(pPaintTime_ms));
	ui->lblDistance->setText(QString::number(pDistance));
}

void MainWindow::SceneObject_Camera(const QString& pName)
{
	ui->lstCamera->addItem(pName);
}

void MainWindow::SceneObject_LightSource(const QString& pName)
{
	ui->lstLight->addItem(pName);
	// After item added "currentRow" is still contain old value (even '-1' if first item added). Because "currentRow"/"currentItem" is changed by user interaction,
	// not by "addItem". So, "currentRow" must be set manually.
	ui->lstLight->setCurrentRow(ui->lstLight->count() - 1);
	// And after "selectAll" handler of "signal itemSelectionChanged" will get right "currentItem" and "currentRow" values.
	ui->lstLight->selectAll();
}

void MainWindow::on_butOpenFile_clicked() {
    aiString filter_temp;
    mImporter.GetExtensionList( filter_temp );

    QString filename, filter;
    filter = filter_temp.C_Str();
	filter.replace(';', ' ');
	filter.append(" ;; All (*.*)");
	filename = QFileDialog::getOpenFileName(this, "Choose the file", "", filter);

    if (!filename.isEmpty()) {
        ImportFile( filename );
    }
}

void MainWindow::on_butExport_clicked()
{
    using namespace Assimp;

#ifndef ASSIMP_BUILD_NO_EXPORT
    QString filename, filter, format_id;
    Exporter exporter;
    QTime time_begin;
    aiReturn rv;
    QStringList exportersList;
    QMap<QString, const aiExportFormatDesc*> exportersMap;


	if(mScene == nullptr)
	{
		QMessageBox::critical(this, "Export error", "Scene is empty");

		return;
	}

	for (size_t i = 0; i < exporter.GetExportFormatCount(); ++i)
	{
		const aiExportFormatDesc* desc = exporter.GetExportFormatDescription(i);
		exportersList.push_back(desc->id + QString(": ") + desc->description);
		exportersMap.insert(desc->id, desc);
	}

	// get an exporter
	bool dialogSelectExporterOk;
	QString selectedExporter = QInputDialog::getItem(this, "Export format", "Select the exporter : ", exportersList, 0, false, &dialogSelectExporterOk);
	if (!dialogSelectExporterOk)
		return;

	// build the filter
	QString selectedId = selectedExporter.left(selectedExporter.indexOf(':'));
	filter = QString("*.") + exportersMap[selectedId]->fileExtension;

	// get file path
	filename = QFileDialog::getSaveFileName(this, "Set file name", "", filter);
	// if it's canceled
	if (filename == "")
		return;

	// begin export
	time_begin = QTime::currentTime();
	rv = exporter.Export(mScene, selectedId.toLocal8Bit(), filename.toLocal8Bit(), aiProcess_FlipUVs);
	ui->lblExportTime->setText(QString::number(time_begin.secsTo(QTime::currentTime())));
	if(rv == aiReturn_SUCCESS)
		LogInfo("Export done: " + filename);
	else
	{
		QString errorMessage = QString("Export failed: ") + filename;
		LogError(errorMessage);
		QMessageBox::critical(this, "Export error", errorMessage);
	}
#endif
}

void MainWindow::on_cbxLighting_clicked(bool pChecked)
{
	if(pChecked)
		mGLView->Lighting_Enable();
	else
		mGLView->Lighting_Disable();

	mGLView->update();
}

void MainWindow::on_lstLight_itemSelectionChanged()
{
	// bool selected = ui->lstLight->isItemSelected(ui->lstLight->currentItem());
	bool selected = ui->lstLight->currentItem()->isSelected();

	if(selected)
		mGLView->Lighting_EnableSource(ui->lstLight->currentRow());
	else
		mGLView->Lighting_DisableSource(ui->lstLight->currentRow());

#if ASSIMP_QT4_VIEWER
	mGLView->updateGL();
#else
	mGLView->update();
#endif // ASSIMP_QT4_VIEWER
}

void MainWindow::on_lstCamera_clicked( const QModelIndex &)
{
	mGLView->Camera_Set(ui->lstLight->currentRow());
#if ASSIMP_QT4_VIEWER
	mGLView->updateGL();
#else
	mGLView->update();
#endif // ASSIMP_QT4_VIEWER
}

void MainWindow::on_cbxBBox_clicked(bool checked)
{
	mGLView->Enable_SceneBBox(checked);
#if ASSIMP_QT4_VIEWER
	mGLView->updateGL();
#else
	mGLView->update();
#endif // ASSIMP_QT4_VIEWER
}

void MainWindow::on_cbxDrawAxes_clicked(bool checked)
{
	mGLView->Enable_Axes(checked);
#if ASSIMP_QT4_VIEWER
	mGLView->updateGL();
#else
	mGLView->update();
#endif // ASSIMP_QT4_VIEWER
}

void MainWindow::on_cbxTextures_clicked(bool checked)
{
	mGLView->Enable_Textures(checked);
	mGLView->update();
}

void MainWindow::closeEvent(QCloseEvent *e)
{
	std::unique_lock<std::mutex> lock(mRayTracingCommandMutex);
	mAssimpOptixRayTracer.quit();
	e->accept();
}