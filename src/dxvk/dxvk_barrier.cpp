#include "dxvk_barrier.h"
#include <assert.h>
#include <unordered_map>
#include <algorithm>
#include <vector>

namespace dxvk {
  
  DxvkBarrierSet:: DxvkBarrierSet(DxvkCmdBuffer cmdBuffer)
  : m_cmdBuffer(cmdBuffer) {

  }


  DxvkBarrierSet::~DxvkBarrierSet() {

  }

  
  void DxvkBarrierSet::accessMemory(
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    m_srcStages |= srcStages;
    m_dstStages |= dstStages;
    
    m_srcAccess |= srcAccess;
    m_dstAccess |= dstAccess;
  }


  void DxvkBarrierSet::accessBuffer(
    const DxvkBufferSliceHandle&    bufSlice,
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    DxvkAccessFlags access = this->getAccessTypes(srcAccess);
    
    if (srcStages == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
     || dstStages == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
      access.set(DxvkAccess::Write);
    
    m_srcStages |= srcStages;
    m_dstStages |= dstStages;
    
    m_srcAccess |= srcAccess;
    m_dstAccess |= dstAccess;

    m_bufSlices.insert(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, access));
  }
  
  
  void DxvkBarrierSet::accessImage(
    const Rc<DxvkImage>&            image,
    const VkImageSubresourceRange&  subresources,
          VkImageLayout             srcLayout,
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          VkImageLayout             dstLayout,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    DxvkAccessFlags access = this->getAccessTypes(srcAccess);

    assert(dstLayout != VK_IMAGE_LAYOUT_UNDEFINED);

    if (srcStages == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
     || dstStages == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
     || srcLayout != dstLayout)
      access.set(DxvkAccess::Write);

    m_srcStages |= srcStages;
    m_dstStages |= dstStages;

    if (srcLayout == dstLayout) {
      m_srcAccess |= srcAccess;
      m_dstAccess |= dstAccess;
    } else {
      VkImageMemoryBarrier barrier;
      barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.pNext                       = nullptr;
      barrier.srcAccessMask               = srcAccess;
      barrier.dstAccessMask               = dstAccess;
      barrier.oldLayout                   = srcLayout;
      barrier.newLayout                   = dstLayout;
      barrier.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
      barrier.image                       = image->handle();
      barrier.subresourceRange            = subresources;
      barrier.subresourceRange.aspectMask = image->formatInfo()->aspectMask;
      m_imgBarriers.push_back(barrier);
    }

    m_imgSlices.insert(image->handle(),
      DxvkBarrierImageSlice(subresources, access));
  }


  void DxvkBarrierSet::releaseBuffer(
          DxvkBarrierSet&           acquire,
    const DxvkBufferSliceHandle&    bufSlice,
          uint32_t                  srcQueue,
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          uint32_t                  dstQueue,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    auto& release = *this;

    release.m_srcStages |= srcStages;
    acquire.m_dstStages |= dstStages;

    VkBufferMemoryBarrier barrier;
    barrier.sType                       = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext                       = nullptr;
    barrier.srcAccessMask               = srcAccess;
    barrier.dstAccessMask               = 0;
    barrier.srcQueueFamilyIndex         = srcQueue;
    barrier.dstQueueFamilyIndex         = dstQueue;
    barrier.buffer                      = bufSlice.handle;
    barrier.offset                      = bufSlice.offset;
    barrier.size                        = bufSlice.length;
    release.m_bufBarriers.push_back(barrier);

    barrier.srcAccessMask               = 0;
    barrier.dstAccessMask               = dstAccess;
    acquire.m_bufBarriers.push_back(barrier);

    DxvkAccessFlags access(DxvkAccess::Read, DxvkAccess::Write);
    release.m_bufSlices.insert(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, access));
    acquire.m_bufSlices.insert(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, access));
  }


  void DxvkBarrierSet::releaseImage(
          DxvkBarrierSet&           acquire,
    const Rc<DxvkImage>&            image,
    const VkImageSubresourceRange&  subresources,
          uint32_t                  srcQueue,
          VkImageLayout             srcLayout,
          VkPipelineStageFlags      srcStages,
          VkAccessFlags             srcAccess,
          uint32_t                  dstQueue,
          VkImageLayout             dstLayout,
          VkPipelineStageFlags      dstStages,
          VkAccessFlags             dstAccess) {
    auto& release = *this;

    release.m_srcStages |= srcStages;
    acquire.m_dstStages |= dstStages;

    VkImageMemoryBarrier barrier;
    barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext                       = nullptr;
    barrier.srcAccessMask               = srcAccess;
    barrier.dstAccessMask               = 0;
    barrier.oldLayout                   = srcLayout;
    barrier.newLayout                   = dstLayout;
    barrier.srcQueueFamilyIndex         = srcQueue;
    barrier.dstQueueFamilyIndex         = dstQueue;
    barrier.image                       = image->handle();
    barrier.subresourceRange            = subresources;
    barrier.subresourceRange.aspectMask = image->formatInfo()->aspectMask;
    release.m_imgBarriers.push_back(barrier);

    if (srcQueue == dstQueue)
      barrier.oldLayout = dstLayout;

    barrier.srcAccessMask               = 0;
    barrier.dstAccessMask               = dstAccess;
    acquire.m_imgBarriers.push_back(barrier);

    DxvkAccessFlags access(DxvkAccess::Read, DxvkAccess::Write);
    release.m_imgSlices.insert(image->handle(),
      DxvkBarrierImageSlice(subresources, access));
    acquire.m_imgSlices.insert(image->handle(),
      DxvkBarrierImageSlice(subresources, access));
  }


  bool DxvkBarrierSet::isBufferDirty(
    const DxvkBufferSliceHandle&    bufSlice,
          DxvkAccessFlags           bufAccess) {
    return m_bufSlices.isDirty(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, bufAccess));
  }


  bool DxvkBarrierSet::isImageDirty(
    const Rc<DxvkImage>&            image,
    const VkImageSubresourceRange&  imgSubres,
          DxvkAccessFlags           imgAccess) {
    return m_imgSlices.isDirty(image->handle(),
      DxvkBarrierImageSlice(imgSubres, imgAccess));
  }


  DxvkAccessFlags DxvkBarrierSet::getBufferAccess(
    const DxvkBufferSliceHandle&    bufSlice) {
    return m_bufSlices.getAccess(bufSlice.handle,
      DxvkBarrierBufferSlice(bufSlice.offset, bufSlice.length, 0));
  }

  
  DxvkAccessFlags DxvkBarrierSet::getImageAccess(
    const Rc<DxvkImage>&            image,
    const VkImageSubresourceRange&  imgSubres) {
    return m_imgSlices.getAccess(image->handle(),
      DxvkBarrierImageSlice(imgSubres, 0));
  }


  void DxvkBarrierSet::recordCommands(const Rc<DxvkCommandList>& commandList) {
    if (m_srcStages | m_dstStages) {
      // AGGRESSIVE BARRIER ELIMINATION: Skip ALL memory-only barriers!
      // Memory-only barriers (from accessMemory/accessBuffer) create 205,031 empty barriers (93%)
      // These are ray tracing sync, buffer tracking, etc. that don't need explicit barriers.
      // ONLY emit barriers when we have actual image or buffer transitions.
      bool hasWork = !m_imgBarriers.empty() || !m_bufBarriers.empty();

      if (!hasWork) {
        // Skip memory-only barriers - no actual image/buffer work
        this->reset();
        return;
      }

      auto tBarrierStart = std::chrono::high_resolution_clock::now();

      VkPipelineStageFlags srcFlags = m_srcStages;
      VkPipelineStageFlags dstFlags = m_dstStages;

      if (!srcFlags) srcFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      if (!dstFlags) dstFlags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

      VkMemoryBarrier memBarrier;
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.pNext = nullptr;
      memBarrier.srcAccessMask = m_srcAccess;
      memBarrier.dstAccessMask = m_dstAccess;

      VkMemoryBarrier* pMemBarrier = nullptr;
      if (m_srcAccess | m_dstAccess)
        pMemBarrier = &memBarrier;

      // DETAILED BARRIER SOURCE TRACKING with PATTERN ANALYSIS
      static uint32_t s_barrierCount = 0;
      static uint32_t s_emptyBarriers = 0;
      static uint32_t s_imageBarriers = 0;
      static uint32_t s_bufferBarriers = 0;

      // Track barrier patterns: key = (srcStages << 32) | dstStages, value = count
      static std::unordered_map<uint64_t, uint32_t> s_stagePatterns;
      static std::unordered_map<uint64_t, uint32_t> s_accessPatterns;
      static std::unordered_map<uint64_t, uint32_t> s_layoutPatterns; // (oldLayout << 16) | newLayout

      s_barrierCount++;

      // Track stage/access patterns
      uint64_t stageKey = (uint64_t(m_srcStages) << 32) | uint64_t(m_dstStages);
      uint64_t accessKey = (uint64_t(m_srcAccess) << 32) | uint64_t(m_dstAccess);
      s_stagePatterns[stageKey]++;
      s_accessPatterns[accessKey]++;

      bool isEmpty = m_imgBarriers.empty() && m_bufBarriers.empty();
      if (isEmpty) {
        s_emptyBarriers++;
        // Log first 10 empty barriers to see the pattern
        if (s_emptyBarriers <= 10) {
          Logger::info(str::format("[BARRIER #", s_barrierCount, "] EMPTY BARRIER: ",
                                  "srcStages=0x", std::hex, m_srcStages,
                                  " dstStages=0x", m_dstStages,
                                  " srcAccess=0x", m_srcAccess,
                                  " dstAccess=0x", m_dstAccess, std::dec));
        }
      } else {
        if (!m_imgBarriers.empty()) {
          s_imageBarriers++;
          // Track layout transition patterns
          for (const auto& imgBarrier : m_imgBarriers) {
            uint64_t layoutKey = (uint64_t(imgBarrier.oldLayout) << 16) | uint64_t(imgBarrier.newLayout);
            s_layoutPatterns[layoutKey]++;
          }
        }
        if (!m_bufBarriers.empty()) s_bufferBarriers++;

        // Log first 10 real barriers
        if ((s_imageBarriers + s_bufferBarriers) <= 10) {
          Logger::info(str::format("[BARRIER #", s_barrierCount, "] ",
                                  m_imgBarriers.size(), " img, ",
                                  m_bufBarriers.size(), " buf"));
          for (size_t i = 0; i < m_imgBarriers.size() && i < 3; ++i) {
            const auto& imgBarrier = m_imgBarriers[i];
            Logger::info(str::format("  [IMG ", i, "] ", imgBarrier.oldLayout, " -> ", imgBarrier.newLayout));
          }
        }
      }

      // Log detailed summary every 10000 barriers showing TOP PATTERNS
      if (s_barrierCount % 10000 == 0) {
        Logger::info(str::format("[BARRIER SUMMARY @ ", s_barrierCount, "] ",
                                "Empty: ", s_emptyBarriers, " (",
                                (s_emptyBarriers * 100 / s_barrierCount), "%), ",
                                "Image: ", s_imageBarriers, ", ",
                                "Buffer: ", s_bufferBarriers));

        // Find top 5 stage patterns
        std::vector<std::pair<uint64_t, uint32_t>> topStages(s_stagePatterns.begin(), s_stagePatterns.end());
        std::sort(topStages.begin(), topStages.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        Logger::info("  TOP STAGE PATTERNS:");
        for (size_t i = 0; i < std::min(size_t(5), topStages.size()); i++) {
          uint32_t src = topStages[i].first >> 32;
          uint32_t dst = topStages[i].first & 0xFFFFFFFF;
          Logger::info(str::format("    [", i+1, "] 0x", std::hex, src, " -> 0x", dst, std::dec,
                                  " (", topStages[i].second, " times)"));
        }

        // Find top 5 layout transition patterns
        if (!s_layoutPatterns.empty()) {
          std::vector<std::pair<uint64_t, uint32_t>> topLayouts(s_layoutPatterns.begin(), s_layoutPatterns.end());
          std::sort(topLayouts.begin(), topLayouts.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
          Logger::info("  TOP LAYOUT TRANSITION PATTERNS:");
          for (size_t i = 0; i < std::min(size_t(5), topLayouts.size()); i++) {
            uint32_t oldLayout = topLayouts[i].first >> 16;
            uint32_t newLayout = topLayouts[i].first & 0xFFFF;
            Logger::info(str::format("    [", i+1, "] ", oldLayout, " -> ", newLayout,
                                    " (", topLayouts[i].second, " times)"));
          }
        }
      }

      commandList->cmdPipelineBarrier(
        m_cmdBuffer, srcFlags, dstFlags, 0,
        pMemBarrier ? 1 : 0, pMemBarrier,
        m_bufBarriers.size(),
        m_bufBarriers.data(),
        m_imgBarriers.size(),
        m_imgBarriers.data());

      commandList->addStatCtr(DxvkStatCounter::CmdBarrierCount, 1);

      this->reset();
    }
  }
  
  
  void DxvkBarrierSet::reset() {
    m_srcStages = 0;
    m_dstStages = 0;

    m_srcAccess = 0;
    m_dstAccess = 0;
    
    m_bufBarriers.resize(0);
    m_imgBarriers.resize(0);

    m_bufSlices.clear();
    m_imgSlices.clear();
  }
  
  
  DxvkAccessFlags DxvkBarrierSet::getAccessTypes(VkAccessFlags flags) {
    const VkAccessFlags rflags
      = VK_ACCESS_INDIRECT_COMMAND_READ_BIT
      | VK_ACCESS_INDEX_READ_BIT
      | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT
      | VK_ACCESS_UNIFORM_READ_BIT
      | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT
      | VK_ACCESS_SHADER_READ_BIT
      | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
      | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
      | VK_ACCESS_TRANSFER_READ_BIT
      | VK_ACCESS_HOST_READ_BIT
      | VK_ACCESS_MEMORY_READ_BIT
      | VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT;
      
    const VkAccessFlags wflags
      = VK_ACCESS_SHADER_WRITE_BIT
      | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_TRANSFER_WRITE_BIT
      | VK_ACCESS_HOST_WRITE_BIT
      | VK_ACCESS_MEMORY_WRITE_BIT
      | VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT
      | VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
    
    DxvkAccessFlags result;
    if (flags & rflags) result.set(DxvkAccess::Read);
    if (flags & wflags) result.set(DxvkAccess::Write);
    return result;
  }
  
}