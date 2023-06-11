#pragma once

#include <string>

namespace manojr
{

/// @brief A class to trace execution of the program, for logs
struct ExecutionCursor
{
    std::string frame;
    int subFrameLevel = 0;

    std::string toString() const {
        return (frame.empty() ? "" : frame + '.') + std::to_string(subFrameLevel);
    }

    ExecutionCursor& advance() {
        ++subFrameLevel;
        return *this;
    }

    ExecutionCursor nextLevel() const {
        ExecutionCursor nextLevelCursor;
        nextLevelCursor.frame = this->toString();
        nextLevelCursor.subFrameLevel = 0;
        return nextLevelCursor;
    }

};

} // namespace manojr
