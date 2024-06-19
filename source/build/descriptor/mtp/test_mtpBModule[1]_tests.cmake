add_test([=[MtpBModuleTest.init]=]  /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/bin/descriptor/mtp/test_mtpBModule [==[--gtest_filter=MtpBModuleTest.init]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[MtpBModuleTest.init]=]  PROPERTIES WORKING_DIRECTORY /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/descriptor/mtp SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_mtpBModule_TESTS MtpBModuleTest.init)
