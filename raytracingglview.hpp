#include "glview.hpp"

#include "assimp/scene.h"

#include <mutex>
#include <thread>


namespace manojr
{

class CRayTracingGLView : public CGLView
{
  public:
    CRayTracingGLView(QWidget *parent, std::mutex& sharedSceneMutex)
    : CGLView(parent),
      sharedSceneMutex_(mSharedSceneMutex)
    {}

  private:
    // Protect shared scene while rendering it from a different (GUI) thread.
    void paintGL() override {
        std::unique_lock<std::mutex> lock(sharedSceneMutex_);
        CGLView::paintGL();
    }

    std::mutex& sharedSceneMutex_;
};

} // namespace manojr
