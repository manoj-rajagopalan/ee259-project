#include <functional>
#include <string>

#include "ExecutionCursor.hpp"

namespace manojr
{

struct RaiiScopeLimitsLogger
{
    RaiiScopeLimitsLogger() = delete;
    RaiiScopeLimitsLogger(std::function<void(std::string const&)> theLogInfo,
                          ExecutionCursor const& theExecutionCursor,
                          std::string&& theScope)
    : logInfo(theLogInfo),
      executionCursor(theExecutionCursor),
      scope(std::move(theScope))
    {
        logInfo('[' + executionCursor.toString() + "]: Begin " + scope);
    }

    ~RaiiScopeLimitsLogger()
    {
        logInfo('[' + executionCursor.toString() + "]: End " + scope);
    }

    std::function<void(std::string const&)> logInfo;
    ExecutionCursor const& executionCursor;
    std::string scope;
};

} // namespace manojr
