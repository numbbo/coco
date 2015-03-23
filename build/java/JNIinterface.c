#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include <jni.h>

#include "coco.h"
#include "coco.c"
#include "JNIinterface.h"

/*
 * Class:     JNIinterface
 * Method:    cocoGetProblem
 * Signature: (Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL Java_JNIinterface_cocoGetProblem
(JNIEnv *jenv, jclass interface_cls, jstring jproblem_suite, jint jfunction_index) {
    
    coco_problem_t *pb = NULL;
    const char *problem_suite;
    
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    problem_suite = (*jenv)->GetStringUTFChars(jenv, jproblem_suite, NULL);
    pb = coco_get_problem(problem_suite, jfunction_index);
    (*jenv)->ReleaseStringUTFChars(jenv, jproblem_suite, problem_suite);
    return (jlong)pb;
}

/*
 * Class:     JNIinterface
 * Method:    cocoObserveProblem
 * Signature: (Ljava/lang/String;JLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_JNIinterface_cocoObserveProblem
(JNIEnv *jenv, jclass interface_cls, jstring jobserver, jlong jproblem, jstring joptions) {
    
    coco_problem_t *pb = NULL;
    const char *observer;
    const char *options;
    
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)jproblem;
    observer = (*jenv)->GetStringUTFChars(jenv, jobserver, NULL);
    options = (*jenv)->GetStringUTFChars(jenv, joptions, NULL);
    pb = coco_observe_problem(observer, pb, options);
    /* Free resources? */
    (*jenv)->ReleaseStringUTFChars(jenv, jobserver, observer);
    /*(*jenv)->ReleaseStringUTFChars(jenv, joptions, options);*/ /* Commented at the moment becuase options is not duplicated in bbob2009_logger() (called by coco_observe_problem()). Has to be enabled however */
    return (jlong)pb;
}

/*
 * Class:     JNIinterface
 * Method:    cocoFreeProblem
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_JNIinterface_cocoFreeProblem
(JNIEnv *jenv, jclass interface_cls, jlong jproblem) {
    
    coco_problem_t *pb = NULL;
    
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)jproblem;
    coco_free_problem(pb);
}

/*
 * Class:     JNIinterface
 * Method:    cocoEvaluateFunction
 * Signature: (LProblem;[D)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_JNIinterface_cocoEvaluateFunction
(JNIEnv *jenv, jclass interface_cls, jobject problem, jdoubleArray x) {

	double *y; /* Result of evaluation. To be allocated with coco_allocate_vector(coco_get_number_of_objectives(pb)) */
	coco_problem_t *pb = NULL; /* Will contain the C problem */
	jint nb_objectives;
    jclass cls;
	jfieldID fid;
	jlong jproblem;
	jdouble *cx;
	jdoubleArray jy; /* Returned double array */

	/* This test is both to prevent warning because interface_cls was not used and check exceptions */
	if (interface_cls == NULL)
		printf("Null interface_cls found\n");
    
    /* Get attributes from jobject problem */
    cls = (*jenv)->GetObjectClass(jenv, problem);
    if (cls == NULL)
        printf("Null cls\n");

	/* Get Problem.problem */
	fid = (*jenv)->GetFieldID(jenv, cls, "problem", "J");
	if(fid == NULL)
		printf("Null fid\n");
	jproblem = (*jenv)->GetLongField(jenv, problem, fid);
	/* Cast it to coco_problem_t */
    pb = (coco_problem_t *)jproblem;

	/* Call coco_evaluate_function */
	cx = (*jenv)->GetDoubleArrayElements(jenv, x, NULL);
    fid = (*jenv)->GetFieldID(jenv, cls, "number_of_objectives", "I");
    if(fid == NULL)
        printf("Null fid2\n");
    nb_objectives = (*jenv)->GetIntField(jenv, problem, fid);
	y = coco_allocate_vector(nb_objectives);
	coco_evaluate_function(pb, cx, y);

	/* Prepare the return value */
	jy = (*jenv)->NewDoubleArray(jenv, nb_objectives);
	(*jenv)->SetDoubleArrayRegion(jenv, jy, 0, nb_objectives, y);

	/* Free resources */
	coco_free_memory(y);
	(*jenv)->ReleaseDoubleArrayElements(jenv, x, cx, JNI_ABORT);
	return jy;
}

/*
 * Class:     JNIinterface
 * Method:    cocoGetNumberOfVariables
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JNIinterface_cocoGetNumberOfVariables
(JNIEnv *jenv, jclass interface_cls, jlong problem) {

	coco_problem_t *pb = NULL;
	jint res;
    jclass cls;

	/* This test is both to prevent warning because interface_cls was not used and check exceptions */
	if (interface_cls == NULL)
		printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    res = coco_get_number_of_variables(pb);
	return res;
}

/*
 * Class:     JNIinterface
 * Method:    cocoGetNumberOfObjectives
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JNIinterface_cocoGetNumberOfObjectives
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    
    coco_problem_t *pb = NULL;
    jint res;
    jclass cls;
    
    /* This test is both to prevent warning because interface_cls was not used and check exceptions */
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    res = coco_get_number_of_objectives(pb);
    return res;
}

/*
 * Class:     JNIinterface
 * Method:    cocoGetSmallestValuesOfInterest
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_JNIinterface_cocoGetSmallestValuesOfInterest
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    
	const double *cres; /* or const jdouble *cres;? */
	coco_problem_t *pb = NULL;
	jint nb_variables;
	jdoubleArray res;
    jclass cls;

	/* This test is both to prevent warning because interface_cls was not used and check exceptions */
	if (interface_cls == NULL)
		printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    cres = coco_get_smallest_values_of_interest(pb);
	nb_variables = coco_get_number_of_variables(pb);

	/* Prepare the return value */
	res = (*jenv)->NewDoubleArray(jenv, nb_variables);
	(*jenv)->SetDoubleArrayRegion(jenv, res, 0, nb_variables, cres);
	return res;
}

/*
 * Class:     JNIinterface
 * Method:    cocoGetLargestValuesOfInterest
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_JNIinterface_cocoGetLargestValuesOfInterest
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    
    const double *cres; /* or const jdouble *cres;? */
    coco_problem_t *pb = NULL;
    jint nb_variables;
    jdoubleArray res;
    jclass cls;
    
    /* This test is both to prevent warning because interface_cls was not used and check exceptions */
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    cres = coco_get_largest_values_of_interest(pb);
    nb_variables = coco_get_number_of_variables(pb);
    
    /* Prepare the return value */
    res = (*jenv)->NewDoubleArray(jenv, nb_variables);
    (*jenv)->SetDoubleArrayRegion(jenv, res, 0, nb_variables, cres);
    return res;
}

/*
 * Class:     JNIinterface
 * Method:    validProblem
 * Signature: (LProblem;)Z
 */
JNIEXPORT jboolean JNICALL Java_JNIinterface_validProblem
(JNIEnv *jenv, jclass interface_cls, jlong jproblem) {
    
    coco_problem_t *pb = NULL;
    jfieldID fid;
    
    /* This test is both to prevent warning because interface_cls was not used and check exceptions */
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)jproblem;
    if (pb == NULL)
        return JNI_FALSE;
    else
        return JNI_TRUE;
}

/*
 * Class:     JNIinterface
 * Method:    cocoGetProblemId
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_JNIinterface_cocoGetProblemId
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    coco_problem_t *pb = NULL;
    const char *res;
    jstring jres;
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    res = coco_get_problem_id(pb);
    jres = (*jenv)->NewStringUTF(jenv, res);
    return jres;
}

/*
 * Class:     JNIinterface
 * Method:    cocoGetProblemName
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_JNIinterface_cocoGetProblemName
(JNIEnv *jenv, jclass interface_cls, jlong jproblem) {
    coco_problem_t *pb = NULL;
    const char *res;
    jstring jres;
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)jproblem;
    res = coco_get_problem_name(pb);
    jres = (*jenv)->NewStringUTF(jenv, res);
    return jres;
}

/*
 * Class:     JNIinterface
 * Method:    cocoGetEvaluations
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JNIinterface_cocoGetEvaluations
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    
    coco_problem_t *pb = NULL;
    jint res;
    jclass cls;
    
    /* This test is both to prevent warning because interface_cls was not used and check exceptions */
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    res = coco_get_evaluations(pb);
    return res;
}

/*
 * Class:     JNIinterface
 * Method:    cocoNextProblemIndex
 * Signature: (Ljava/lang/String;ILjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_JNIinterface_cocoNextProblemIndex
(JNIEnv *jenv, jclass interface_cls, jstring jproblem_suite, jint problem_index, jstring jselect_options) {
    
    const char *problem_suite;
    const char *select_options;
    jint res;
    
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    problem_suite = (*jenv)->GetStringUTFChars(jenv, jproblem_suite, NULL);
    select_options = (*jenv)->GetStringUTFChars(jenv, jselect_options, NULL);
    res = coco_next_problem_index(problem_suite, problem_index, select_options);
    /* Free resources */
    (*jenv)->ReleaseStringUTFChars(jenv, jproblem_suite, problem_suite);
    (*jenv)->ReleaseStringUTFChars(jenv, jselect_options, select_options);
    return res;
}

