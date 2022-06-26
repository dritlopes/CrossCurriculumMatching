import copy
from utils import find_age

def get_search_space (data, filters, target_curriculum):

    """
    Filter target curriculum and other unwanted curricula from search space and filter grades and subjects if specified

    :param data: dict with curriculum trees
    :param filters: dict with filters on curriculums
    :param target_curriculum: the curriculum to be matched
    :return: filtered dictionary of curriculum trees
    """

    data_filtered = copy.deepcopy(data)

    for cur_id, cur in data.items():
        if filters['curriculums'] and cur['label'] not in filters['curriculums']:
            data_filtered.pop(cur_id)
            continue
        elif cur['label'] == target_curriculum:
            data_filtered.pop(cur_id)
            continue

        for grade_id, grade in cur['grade'].items():
            if filters['grades'] and grade['label'] not in filters['grades']:
                data_filtered[cur_id]['grade'].pop(grade_id)
                continue
            # if target_grade and filterAge and age_to_grade:
            #     grades_to_include = filter_grade(target_grade, target_curriculum, age_to_grade)
            #     if grade_id not in grades_to_include:
            #         data_filtered[cur_id]['grade'].pop(grade_id)

            for subject_id, subject in grade['subject'].items():
                if filters['subjects'] and subject['label'] not in filters['subjects']:
                    data_filtered[cur_id]['grade'][grade_id]['subject'].pop(subject_id)
                    continue

    return data_filtered

# def filter_grade (target_grade, target_curriculum, age_to_grade):
#
#     """
#     Find grades to include according to age targeted by the queries to be matched.
#     :param target_grade: grade of queries to be matched
#     :param target_curriculum: curriculum of queries to be matched
#     :param age_to_grade: dict with age as key and grades as value
#     :return: list of grades to include in the search space
#     """
#
#     grades_to_include = []
#     target_age = find_age(age_to_grade,target_curriculum,target_grade)
#
#     # when target grade is not in 'Reading levels and age filter settings' file, filter age will have no effect
#     if target_age == -1: grades_to_include = None
#
#     else:
#         for age in [target_age, target_age - 1, target_age + 1, target_age + 2, target_age - 2]:
#             if age in age_to_grade.keys():
#                 for grade in age_to_grade[age]:
#                     if grade['CURRICULUM'] != target_curriculum: grades_to_include.append(grade['GRADEID'])
#
#     return grades_to_include