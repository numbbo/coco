/**
 * This file contains all necessary interfaces to the COCO code in C. The structures coco_problem_s,
 * coco_suite_s and coco_observer_s are accessed by means of "pointers" of type long.
 *
 * TODO: Check if the casts from pointer to C structure actually work (how can this be done?)
 */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include <jni.h>

#include "coco.h"
#include "coco.c"
#include "CocoJNI.h"

/*
 * Class:     CocoJNI
 * Method:    cocoSetLogLevel
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_CocoJNI_cocoSetLogLevel
(JNIEnv *jenv, jclass interface_cls, jstring jlog_level) {

  const char *log_level;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoSetLogLevel\n");
  }

  log_level = (*jenv)->GetStringUTFChars(jenv, jlog_level, NULL);

  coco_set_log_level(log_level);

  return;
}

/*
 * Class:     CocoJNI
 * Method:    cocoGetObserver
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoGetObserver
(JNIEnv *jenv, jclass interface_cls, jstring jobserver_name, jstring jobserver_options) {

  coco_observer_t *observer = NULL;
  const char *observer_name;
  const char *observer_options;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoGetObserver\n");
  }

  observer_name = (*jenv)->GetStringUTFChars(jenv, jobserver_name, NULL);
  observer_options = (*jenv)->GetStringUTFChars(jenv, jobserver_options, NULL);

  observer = coco_observer(observer_name, observer_options);

  return (jlong) observer;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemAddObserver
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoProblemAddObserver
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer, jlong jobserver_pointer) {

  coco_observer_t *observer = NULL;
  coco_problem_t *problem_before = NULL;
  coco_problem_t *problem_after = NULL;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemAddObserver\n");
  }

  observer = (coco_observer_t *) jobserver_pointer;
  problem_before = (coco_problem_t *) jproblem_pointer;

  problem_after = coco_problem_add_observer(problem_before, observer);

  return (jlong) problem_after;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemRemoveObserver
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoProblemRemoveObserver
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer, jlong jobserver_pointer) {

  coco_observer_t *observer = NULL;
  coco_problem_t *problem_before = NULL;
  coco_problem_t *problem_after = NULL;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemRemoveObserver\n");
  }

  observer = (coco_observer_t *) jobserver_pointer;
  problem_before = (coco_problem_t *) jproblem_pointer;

  problem_after = coco_problem_remove_observer(problem_before, observer);

  return (jlong) problem_after;
}

/*
 * Class:     CocoJNI
 * Method:    cocoFinalizeObserver
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_CocoJNI_cocoFinalizeObserver
(JNIEnv *jenv, jclass interface_cls, jlong jobserver_pointer) {

  coco_observer_t *observer = NULL;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoFinalizeObserver\n");
  }

  observer = (coco_observer_t *) jobserver_pointer;
  coco_observer_free(observer);
  return;
}

/*
 * Class:     CocoJNI
 * Method:    Java_CocoJNI_cocoObserverSignalRestart
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_CocoJNI_cocoObserverSignalRestart
(JNIEnv *jenv, jclass interface_cls, jlong jobserver_pointer, jlong jproblem_pointer) {

  coco_observer_t *observer = NULL;
  coco_problem_t *problem = NULL;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in Java_CocoJNI_cocoObserverSignalRestart\n");
  }

  observer = (coco_observer_t *) jobserver_pointer;
  problem = (coco_problem_t *) jproblem_pointer;

  coco_observer_signal_restart(observer, problem);

}

/*
 * Class:     CocoJNI
 * Method:    cocoGetSuite
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoGetSuite
(JNIEnv *jenv, jclass interface_cls, jstring jsuite_name, jstring jsuite_instance, jstring jsuite_options) {

  coco_suite_t *suite = NULL;
  const char *suite_name;
  const char *suite_instance;
  const char *suite_options;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoGetSuite\n");
  }

  suite_name = (*jenv)->GetStringUTFChars(jenv, jsuite_name, NULL);
  suite_instance = (*jenv)->GetStringUTFChars(jenv, jsuite_instance, NULL);
  suite_options = (*jenv)->GetStringUTFChars(jenv, jsuite_options, NULL);

  suite = coco_suite(suite_name, suite_instance, suite_options);

  return (jlong) suite;
}

/*
 * Class:     CocoJNI
 * Method:    cocoFinalizeSuite
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_CocoJNI_cocoFinalizeSuite
(JNIEnv *jenv, jclass interface_cls, jlong jsuite_pointer) {

  coco_suite_t *suite = NULL;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoFinalizeSuite\n");
  }

  suite = (coco_suite_t *) jsuite_pointer;
  coco_suite_free(suite);
  return;
}

/*
 * Class:     CocoJNI
 * Method:    cocoSuiteGetNumberOfProblems
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoSuiteGetNumberOfProblems
(JNIEnv *jenv, jclass interface_cls, jlong jsuite_pointer) {

  coco_suite_t *suite = NULL;
  jlong jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoSuiteGetNumberOfProblems\n");
  }

  suite = (coco_suite_t *) jsuite_pointer;
  jresult = (jlong) coco_suite_get_number_of_problems(suite);

  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoSuiteGetNextProblem
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoSuiteGetNextProblem
(JNIEnv *jenv, jclass interface_cls, jlong jsuite_pointer, jlong jobserver_pointer) {

  coco_problem_t *problem = NULL;
  coco_suite_t *suite = NULL;
  coco_observer_t *observer = NULL;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoSuiteGetNextProblem\n");
  }

  suite = (coco_suite_t *) jsuite_pointer;
  observer = (coco_observer_t *) jobserver_pointer;
  problem = coco_suite_get_next_problem(suite, observer);

  if (problem == NULL)
    return 0;

  return (jlong) problem;
}

/*
 * Class:     CocoJNI
 * Method:    cocoSuiteGetProblem
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoSuiteGetProblem
(JNIEnv *jenv, jclass interface_cls, jlong jsuite_pointer, jlong jproblem_index) {

  coco_problem_t *problem = NULL;
  coco_suite_t *suite = NULL;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoSuiteGetProblem\n");
  }

  suite = (coco_suite_t *) jsuite_pointer;
  problem = coco_suite_get_problem(suite, jproblem_index);

  if (problem == NULL)
    return 0;

  return (jlong) problem;
}

/*
 * Class:     CocoJNI
 * Method:    cocoSuiteGetProblemByFuncDimInst
 * Signature: (JJJJ)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoSuiteGetProblemByFuncDimInst
(JNIEnv *jenv, jclass interface_cls, jlong jsuite_pointer, jlong jfunction, jlong jdimension, jlong jinstance) {

  coco_problem_t *problem = NULL;
  coco_suite_t *suite = NULL;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoSuiteGetProblemByFuncDimInst\n");
  }

  suite = (coco_suite_t *) jsuite_pointer;
  problem = coco_suite_get_problem_by_function_dimension_instance(suite, jfunction, jdimension, jinstance);

  if (problem == NULL)
    return 0;

  return (jlong) problem;
}

/*
 * Class:     CocoJNI
 * Method:    cocoEvaluateFunction
 * Signature: (J[D)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_CocoJNI_cocoEvaluateFunction
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer, jdoubleArray jx) {

  coco_problem_t *problem = NULL;
  double *y = NULL;
  double *x = NULL;
  int number_of_objectives;
  jdoubleArray jy;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoEvaluateFunction\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  number_of_objectives = (int) coco_problem_get_number_of_objectives(problem);

  /* Call coco_evaluate_function */
  x = (*jenv)->GetDoubleArrayElements(jenv, jx, NULL);
  y = coco_allocate_vector(number_of_objectives);
  coco_evaluate_function(problem, x, y);

  /* Prepare the return value */
  jy = (*jenv)->NewDoubleArray(jenv, number_of_objectives);
  (*jenv)->SetDoubleArrayRegion(jenv, jy, 0, number_of_objectives, y);

  /* Free resources */
  coco_free_memory(y);
  (*jenv)->ReleaseDoubleArrayElements(jenv, jx, x, JNI_ABORT);
  return jy;
}

/*
 * Class:     CocoJNI
 * Method:    cocoEvaluateConstraint
 * Signature: (J[D)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_CocoJNI_cocoEvaluateConstraint
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer, jdoubleArray jx) {

  coco_problem_t *problem = NULL;
  double *y = NULL;
  double *x = NULL;
  int number_of_constraints;
  jdoubleArray jy;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoEvaluateConstraint\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  number_of_constraints = (int) coco_problem_get_number_of_constraints(problem);

  /* Call coco_evaluate_constraint */
  x = (*jenv)->GetDoubleArrayElements(jenv, jx, NULL);
  y = coco_allocate_vector(number_of_constraints);
  coco_evaluate_constraint(problem, x, y);

  /* Prepare the return value */
  jy = (*jenv)->NewDoubleArray(jenv, number_of_constraints);
  (*jenv)->SetDoubleArrayRegion(jenv, jy, 0, number_of_constraints, y);

  /* Free resources */
  coco_free_memory(y);
  (*jenv)->ReleaseDoubleArrayElements(jenv, jx, x, JNI_ABORT);
  return jy;
}

/*
 * Class:     CocoJNI
 * Method:    cocoRecommendSolution
 * Signature: (J[D)V
 */
JNIEXPORT void JNICALL Java_CocoJNI_cocoRecommendSolution
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer, jdoubleArray jx) {

  coco_problem_t *problem = NULL;
  double *x = NULL;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoRecommendSolution\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;

  /* Call coco_evaluate_constraint */
  x = (*jenv)->GetDoubleArrayElements(jenv, jx, NULL);
  coco_recommend_solution(problem, x);

  /* Free resources */
  (*jenv)->ReleaseDoubleArrayElements(jenv, jx, x, JNI_ABORT);
  return;
}


/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetDimension
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_CocoJNI_cocoProblemGetDimension
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  jint jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetDimension\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  jresult = (jint) coco_problem_get_dimension(problem);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetNumberOfObjectives
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_CocoJNI_cocoProblemGetNumberOfObjectives
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  jint jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetNumberOfObjectives\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  jresult = (jint) coco_problem_get_number_of_objectives(problem);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetNumberOfConstraints
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_CocoJNI_cocoProblemGetNumberOfConstraints
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  jint jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetNumberOfConstraints\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  jresult = (jint) coco_problem_get_number_of_constraints(problem);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetSmallestValuesOfInterest
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_CocoJNI_cocoProblemGetSmallestValuesOfInterest
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  const double *result;
  jdoubleArray jresult;
  jint dimension;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetSmallestValuesOfInterest\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  dimension = (int) coco_problem_get_dimension(problem);
  result = coco_problem_get_smallest_values_of_interest(problem);

  /* Prepare the return value */
  jresult = (*jenv)->NewDoubleArray(jenv, dimension);
  (*jenv)->SetDoubleArrayRegion(jenv, jresult, 0, dimension, result);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetLargestValuesOfInterest
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_CocoJNI_cocoProblemGetLargestValuesOfInterest
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  const double *result;
  jdoubleArray jresult;
  jint dimension;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetLargestValuesOfInterest\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  dimension = (int) coco_problem_get_dimension(problem);
  result = coco_problem_get_largest_values_of_interest(problem);

  /* Prepare the return value */
  jresult = (*jenv)->NewDoubleArray(jenv, dimension);
  (*jenv)->SetDoubleArrayRegion(jenv, jresult, 0, dimension, result);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetNumberOfIntegerVariables
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_CocoJNI_cocoProblemGetNumberOfIntegerVariables
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  jint jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetNumberOfIntegerVariables\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  jresult = (jint) coco_problem_get_number_of_integer_variables(problem);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetLargestFValuesOfInterest
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_CocoJNI_cocoProblemGetLargestFValuesOfInterest
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  const double *result;
  jdoubleArray jresult;
  jint num_obj;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetLargestFValuesOfInterest\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  num_obj = (int) coco_problem_get_number_of_objectives(problem);
  if (num_obj == 1)
  	return NULL;
  result = coco_problem_get_largest_fvalues_of_interest(problem);

  /* Prepare the return value */
  jresult = (*jenv)->NewDoubleArray(jenv, num_obj);
  (*jenv)->SetDoubleArrayRegion(jenv, jresult, 0, num_obj, result);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetId
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_CocoJNI_cocoProblemGetId
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  const char *result;
  jstring jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetId\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  result = coco_problem_get_id(problem);
  jresult = (*jenv)->NewStringUTF(jenv, result);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetName
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_CocoJNI_cocoProblemGetName
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  const char *result;
  jstring jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetName\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  result = coco_problem_get_name(problem);
  jresult = (*jenv)->NewStringUTF(jenv, result);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetEvaluations
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoProblemGetEvaluations
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  jlong jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetEvaluations\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  jresult = (jlong) coco_problem_get_evaluations(problem);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetEvaluationsConstraints
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoProblemGetEvaluationsConstraints
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {
    
  coco_problem_t *problem = NULL;
  jlong jresult;
    
  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetEvaluationsConstraints\n");
  }
    
  problem = (coco_problem_t *) jproblem_pointer;
  jresult = (jlong) coco_problem_get_evaluations_constraints(problem);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemGetIndex
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_CocoJNI_cocoProblemGetIndex
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  jlong jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetIndex\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  jresult = (jlong) coco_problem_get_suite_dep_index(problem);
  return jresult;
}

/*
 * Class:     CocoJNI
 * Method:    cocoProblemIsFinalTargetHit
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_CocoJNI_cocoProblemIsFinalTargetHit
(JNIEnv *jenv, jclass interface_cls, jlong jproblem_pointer) {

  coco_problem_t *problem = NULL;
  jint jresult;

  /* This test is both to prevent warning because interface_cls was not used and to check for exceptions */
  if (interface_cls == NULL) {
    jclass Exception = (*jenv)->FindClass(jenv, "java/lang/Exception");
    (*jenv)->ThrowNew(jenv, Exception, "Exception in cocoProblemGetIndex\n");
  }

  problem = (coco_problem_t *) jproblem_pointer;
  jresult = (jint) coco_problem_final_target_hit(problem);
  return jresult;
}
