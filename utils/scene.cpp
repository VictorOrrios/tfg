

#include "scene.hpp"

Scene::Scene(){
    m_root.text = "Root";

    auto nodeA = std::make_unique<Node>();
    nodeA->text = "A";

    auto nodeB = std::make_unique<Node>();
    nodeB->text = "B";

    m_root.children.push_back(std::move(nodeA));
    m_root.children.push_back(std::move(nodeB));
}

void Scene::draw(){ 
    drawNode(&m_root); 
}

void Scene::drawNode(Scene::Node* node){
    if (ImGui::TreeNodeEx(node->text.c_str(),m_flags)){
        for (auto&& child : node->children)
            drawNode(child.get());

        ImGui::TreePop();
    }
}