project('coco', 'c',
  default_options: ['warning_level=3', 'buildtype=release']
  )

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : false)

coco_lib = static_library('coco', 
  sources: 'coco.c',
  dependencies: m_dep
  )

executable('example_experiment', 
  sources: 'example_experiment.c',
  link_with: coco_lib,
  dependencies: m_dep
  )

executable('test_coco', 
  sources: 'test_coco.c',
  link_with: coco_lib,
  dependencies: m_dep
  )

