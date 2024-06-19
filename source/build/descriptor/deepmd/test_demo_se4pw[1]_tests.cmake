add_test([=[DemoSe4pwTest.demo]=]  /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/bin/descriptor/deepmd/test_demo_se4pw [==[--gtest_filter=DemoSe4pwTest.demo]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[DemoSe4pwTest.demo]=]  PROPERTIES WORKING_DIRECTORY /data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/source/build/descriptor/deepmd SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_demo_se4pw_TESTS DemoSe4pwTest.demo)
