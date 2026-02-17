#pragma once

#include <string>
#include <vector>
#include <memory>
#include <imgui.h>


class Scene{
public:
    struct Node
    {
        std::string text;
        std::vector<std::unique_ptr<Node>> children;
    };

    Scene();

    void draw();

private:
    void drawNode(Node* node);

    Node m_root;

    ImGuiTreeNodeFlags m_flags =
    ImGuiTreeNodeFlags_OpenOnArrow |
    ImGuiTreeNodeFlags_OpenOnDoubleClick |
    ImGuiTreeNodeFlags_DefaultOpen;
};