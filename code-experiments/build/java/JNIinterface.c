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
 * Method:    cocoSuiteGetProblem
 * Signature: (Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL Java_JNIinterface_cocoSuiteGetProblem
(JNIEnv *jenv, jclass interface_cls, jstring jproblem_suite, jlong jfunction_index) {
    
    coco_problem_t *pb = NULL;
    const char *problem_suite;
    
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    problem_suite = (*jenv)->GetStringUTFChars(jenv, jproblem_suite, NULL);
    pb = coco_suite_get_problem(problem_suite, jfunction_index);
    (*jenv)->ReleaseStringUTFChars(jenv, jproblem_suite, problem_suite);
    return (jlong)pb;
}

/*
 * Class:     JNIinterface
 * Method:    cocoProblemAddObserverDeprecated
 * Signature: (Ljava/lang/String;JLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_JNIinterface_cocoProblemAddObserverDeprecated
(JNIEnv *jenv, jclass interface_cls, jlong jproblem, jstring jobserver, jstring joptions) {
    
    coco_problem_t *pb = NULL;
    const char *observer;
    const char *options;
    
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)jproblem;
    observer = (*jenv)->GetStringUTFChars(jenv, jobserver, NULL);
    options = (*jenv)->GetStringUTFChars(jenv, joptions, NULL);
    pb = deprecated__coco_problem_add_observer(pb, observer, options);
    /* Free resources? */
    (*jenv)->ReleaseStringUTFChars(jenv, jobserver, observer);
    /*(*jenv)->ReleaseStringUTFChars(jenv, joptions, options);*/ /* Commented at the moment becuase options is not duplicated in logger_bbob2009() (called by coco_problem_add_observer()). Has to be enabled however */
    return (jlong)pb;
}

/*
 * Class:     JNIinterface
 * Method:    cocoProblemFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_JNIinterface_cocoProblemFree
(JNIEnv *jenv, jclass interface_cls, jlong jproblem) {
    
    coco_problem_t *pb = NULL;
    
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)jproblem;
    coco_problem_free(pb);
}

/*
 * Class:     JNIinterface
 * Method:    cocoEvaluateFunction
 * Signature: (LProblem;[D)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_JNIinterface_cocoEvaluateFunction
(JNIEnv *jenv, jclass interface_cls, jobject problem, jdoubleArray x) {

	double *y; /* Result of evaluation. To be allocated with coco_allocate_vector(coco_problem_get_number_of_objectives(pb)) */
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
 * Method:    cocoProblemGetDimension
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JNIinterface_cocoProblemGetDimension
(JNIEnv *jenv, jclass interface_cls, jlong problem) {

	coco_problem_t *pb = NULL;
	jint res;
    jclass cls;

	/* This test is both to prevent warning because interface_cls was not used and check exceptions */
	if (interface_cls == NULL)
		printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    res = coco_problem_get_dimension(pb);
	return res;
}

/*
 * Class:     JNIinterface
 * Method:    cocoProblemGetNumberOfObjectives
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JNIinterface_cocoProblemGetNumberOfObjectives
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    
    coco_problem_t *pb = NULL;
    jint res;
    jclass cls;
    
    /* This test is both to prevent warning because interface_cls was not used and check exceptions */
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    res = coco_problem_get_number_of_objectives(pb);
    return res;
}

/*
 * Class:     JNIinterface
 * Method:    cocoProblemGetSmallestValuesOfInterest
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_JNIinterface_cocoProblemGetSmallestValuesOfInterest
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    
	const double *cres; /* or const jdouble *cres;? */
	coco_problem_t *pb = NULL;
	jint nb_dim;
	jdoubleArray res;
    jclass cls;

	/* This test is both to prevent warning because interface_cls was not used and check exceptions */
	if (interface_cls == NULL)
		printf("Null interface_cls found\n");

	pb = (coco_problem_t *)problem;
  cres = coco_problem_get_smallest_values_of_interest(pb);
	nb_dim = coco_problem_get_dimension(pb);

	/* Prepare the return value */
	res = (*jenv)->NewDoubleArray(jenv, nb_dim);
	(*jenv)->SetDoubleArrayRegion(jenv, res, 0, nb_dim, cres);
	return res;
}

/*
 * Class:     JNIinterface
 * Method:    cocoProblemGetLargestValuesOfInterest
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_JNIinterface_cocoProblemGetLargestValuesOfInterest
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    
    const double *cres; /* or const jdouble *cres;? */
    coco_problem_t *pb = NULL;
    jint nb_dim;
    jdoubleArray res;
    jclass cls;
    
    /* This test is both to prevent warning because interface_cls was not used and check exceptions */
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    cres = coco_problem_get_largest_values_of_interest(pb);
    nb_dim = coco_problem_get_dimension(pb);
    
    /* Prepare the return value */
    res = (*jenv)->NewDoubleArray(jenv, nb_dim);
    (*jenv)->SetDoubleArrayRegion(jenv, res, 0, nb_dim, cres);
    return res;
}

/*
 * Class:     JNIinterface
 * Method:    cocoProblemIsValid
 * Signature: (LProblem;)Z
 */
JNIEXPORT jboolean JNICALL Java_JNIinterface_cocoProblemIsValid
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
 * Method:    cocoProblemGetId
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_JNIinterface_cocoProblemGetId
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    coco_problem_t *pb = NULL;
    const char *res;
    jstring jres;
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    res = coco_problem_get_id(pb);
    jres = (*jenv)->NewStringUTF(jenv, res);
    return jres;
}

/*
 * Class:     JNIinterface
 * Method:    cocoProblemGetName
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_JNIinterface_cocoProblemGetName
(JNIEnv *jenv, jclass interface_cls, jlong jproblem) {
    coco_problem_t *pb = NULL;
    const char *res;
    jstring jres;
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)jproblem;
    res = coco_problem_get_name(pb);
    jres = (*jenv)->NewStringUTF(jenv, res);
    return jres;
}

/*
 * Class:     JNIinterface
 * Method:    cocoProblemGetEvaluations
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_JNIinterface_cocoProblemGetEvaluations
(JNIEnv *jenv, jclass interface_cls, jlong problem) {
    
    coco_problem_t *pb = NULL;
    jint res;
    jclass cls;
    
    /* This test is both to prevent warning because interface_cls was not used and check exceptions */
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    pb = (coco_problem_t *)problem;
    res = coco_problem_get_evaluations(pb);
    return res;
}

/*
 * Class:     JNIinterface
 * Method:    cocoSuiteGetNextProblemIndex
 * Signature: (Ljava/lang/String;ILjava/lang/String;)I
 */
JNIEXPORT jlong JNICALL Java_JNIinterface_cocoSuiteGetNextProblemIndex
(JNIEnv *jenv, jclass interface_cls, jstring jproblem_suite, jlong problem_index, jstring jselect_options) {
    
    const char *problem_suite;
    const char *select_options;
    jint res;
    
    if (interface_cls == NULL)
        printf("Null interface_cls found\n");
    problem_suite = (*jenv)->GetStringUTFChars(jenv, jproblem_suite, NULL);
    select_options = (*jenv)->GetStringUTFChars(jenv, jselect_options, NULL);
    res = coco_suite_get_next_problem_index(problem_suite, problem_index, select_options);
    /* Free resources */
    (*jenv)->ReleaseStringUTFChars(jenv, jproblem_suite, problem_suite);
    (*jenv)->ReleaseStringUTFChars(jenv, jselect_options, select_options);
    return res;
}

