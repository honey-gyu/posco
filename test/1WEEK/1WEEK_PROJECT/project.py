#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

def show(slist):
    if not slist:
        print("List is empty.")
        return

    for student in slist:
        student['average'] = (student['score1'] + student['score2']) / 2
        if student['average'] >= 90:
            student['grade'] = 'A'
        elif student['average'] >= 80:
            student['grade'] = 'B'
        elif student['average'] >= 70:
            student['grade'] = 'C'
        elif student['average'] >= 60:
            student['grade'] = 'D'
        else:
            student['grade'] = 'F'
    
    sorted_list = sorted(slist, key=lambda x: x['average'], reverse=True)
    
    print(f"{'Student':>12} {'Name':>15} {'Midterm':^8} {'Final':^6} {'Average':^8} {'Grade':^6}")
    print("-------------------------------------------------------------")
    for student in sorted_list:
        print(f"{student['id']:>12} {student['name']:>15} {student['score1']:^8} {student['score2']:^6} {student['average']:^8.1f} {student['grade']:^6}")

student_list = [
    {'id': '20180001', 'name': 'Hong Gildong', 'score1': 84, 'score2': 73, 'average': 78.5, 'grade': 'C'},
    {'id': '20180002', 'name': 'Lee Jieun', 'score1': 92, 'score2': 89, 'average': 90.5, 'grade': 'A'},
    {'id': '20180007', 'name': 'Kim Cheolsu', 'score1': 57, 'score2': 62, 'average': 59.5, 'grade': 'F'},
    {'id': '20180009', 'name': 'Lee Yeonghee', 'score1': 81, 'score2': 84, 'average': 82.5, 'grade': 'B'},
    {'id': '20180011', 'name': 'Ha Donghun', 'score1': 58, 'score2': 68, 'average': 63.0, 'grade': 'D'}
]

def search(slist):
    while True:
        student_id = input("Student ID: ").strip()
        found = False
        for student in slist:
            if student['id'] == student_id:
                print(f"{'Student':>12} {'Name':>15} {'Midterm':^8} {'Final':^6} {'Average':^8} {'Grade':^6}")
                print("-------------------------------------------------------------")
                print(f"{student['id']:>12} {student['name']:>15} {student['score1']:^8} {student['score2']:^6} {student['average']:^8.1f} {student['grade']:^6}")
                found = True
                break
        if not found:
            print("NO SUCH PERSON.")
        else:
            break

student_list = [
    {'id': '20180001', 'name': 'Hong Gildong', 'score1': 84, 'score2': 73, 'average': 78.5, 'grade': 'C'},
    {'id': '20180002', 'name': 'Lee Jieun', 'score1': 92, 'score2': 89, 'average': 90.5, 'grade': 'A'},
    {'id': '20180007', 'name': 'Kim Cheolsu', 'score1': 57, 'score2': 62, 'average': 59.5, 'grade': 'F'},
    {'id': '20180009', 'name': 'Lee Yeonghee', 'score1': 81, 'score2': 84, 'average': 82.5, 'grade': 'B'},
    {'id': '20180011', 'name': 'Ha Donghun', 'score1': 58, 'score2': 68, 'average': 63.0, 'grade': 'D'}
]
            
def changeScore(slist):
    while True:
        student_id = input("Student ID: ").strip()
        student = next((s for s in slist if s['id'] == student_id), None)
        if not student:
            print("NO SUCH PERSON.")
            continue
        
        while True:
            score_type = input("Mid/Final: ").strip().lower()
            if score_type in ['mid', 'final']:
                current_score = student['score1'] if score_type == 'mid' else student['score2']
                print(f"Current {score_type} score: {current_score}")
                while True:
                    new_score = input("Input new score: ").strip()
                    try:
                        new_score = int(new_score)
                        if 0 <= new_score <= 100:
                            if score_type == 'mid':
                                student['score1'] = new_score
                            else:
                                student['score2'] = new_score
                            student['average'] = (student['score1'] + student['score2']) / 2
                            if student['average'] >= 90:
                                student['grade'] = 'A'
                            elif student['average'] >= 80:
                                student['grade'] = 'B'
                            elif student['average'] >= 70:
                                student['grade'] = 'C'
                            elif student['average'] >= 60:
                                student['grade'] = 'D'
                            else:
                                student['grade'] = 'F'
                            print(f"{'Student':>12} {'Name':>15} {'Midterm':^8} {'Final':^6} {'Average':^8} {'Grade':^6}")
                            print("-------------------------------------------------------------")
                            print(f"{student['id']:>12} {student['name']:>15} {current_score:^8} {student['score2']:^6} {student['average']:^8.1f} {student['grade']:^6}")
                            print("Score changed:")
                            print(f"{student['id']:>12} {student['name']:>15} {student['score1']:^8} {student['score2']:^6} {student['average']:^8.1f} {student['grade']:^6}")
                            break
                        else:
                            print("")
                    except ValueError:
                        print("")
                break
            else:
                print("")
        break

def searchGrade(slist):
    while True:
        grade = input("Grade to search: ").strip().upper()
        if grade not in ['A', 'B', 'C', 'D', 'F']:
            print("")
            continue
        
        results = [s for s in slist if s['grade'] == grade]
        if results:
            print(f"{'Student':>12} {'Name':>15} {'Midterm':^8} {'Final':^6} {'Average':^8} {'Grade':^6}")
            print("-------------------------------------------------------------")
            for student in results:
                print(f"{student['id']:>12} {student['name']:>15} {student['score1']:^8} {student['score2']:^6} {student['average']:^8.1f} {student['grade']:^6}")
        else:
            print("NO RESULTS.")
        break

student_list = [
    {'id': '20180001', 'name': 'Hong Gildong', 'score1': 84, 'score2': 73, 'average': 78.5, 'grade': 'C'},
    {'id': '20180002', 'name': 'Lee Jieun', 'score1': 92, 'score2': 89, 'average': 90.5, 'grade': 'A'},
    {'id': '20180007', 'name': 'Kim Cheolsu', 'score1': 57, 'score2': 62, 'average': 59.5, 'grade': 'F'},
    {'id': '20180009', 'name': 'Lee Yeonghee', 'score1': 81, 'score2': 84, 'average': 82.5, 'grade': 'B'},
    {'id': '20180011', 'name': 'Ha Donghun', 'score1': 58, 'score2': 68, 'average': 63.0, 'grade': 'D'}
]

def add(slist):
    while True:
        student_id = input("Student ID: ").strip()
        if any(s['id'] == student_id for s in slist):
            print("ALREADY EXISTS.")
            continue

        name = input("Name: ").strip()
        if not name:
            print("")
            continue
        
        try:
            score1 = int(input("Midterm Score: ").strip())
            score2 = int(input("Final Score: ").strip())
            if 0 <= score1 <= 100 and 0 <= score2 <= 100:
                average = (score1 + score2) / 2
                if average >= 90:
                    grade = 'A'
                elif average >= 80:
                    grade = 'B'
                elif average >= 70:
                    grade = 'C'
                elif average >= 60:
                    grade = 'D'
                else:
                    grade = 'F'
                
                slist.append({
                    'id': student_id,
                    'name': name,
                    'score1': score1,
                    'score2': score2,
                    'average': average,
                    'grade': grade
                })
                print("Student added.")
                break
            else:
                print("")
        except ValueError:
            print("")

def remove(slist):
    if not slist:
        print("List is empty.")
        return

    while True:
        student_id = input("Student ID: ").strip()
        student = next((s for s in slist if s['id'] == student_id), None)
        if student:
            slist.remove(student)
            print("Student removed.")
            break
        else:
            print("NO SUCH PERSON.")
            break

def quit(slist):
    while True:
        user_input = input("Save data?[yes/no]: ").strip().lower()

        if user_input == 'yes':
            while True:
                filename = input("File name: ").strip()
                
                if not filename:
                    filename = 'students.txt'
                    break
                
                if ' ' in filename:
                    print("")
                else:
                    break
            
            sorted_list = sorted(slist, key=lambda x: x['average'], reverse=True)
            
            try:
                with open(filename, 'w') as file:
                    for student in sorted_list:
                        file.write(f"{student['id']}\t{student['name']}\t{student['score1']}\t{student['score2']}\n")
            except Exception as e:
                print("")
            
            break

        elif user_input == 'no':
            break

        else:
            print("")

def load_students(filename):
    slist = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    student_id, name, score1, score2 = parts
                    score1 = int(float(score1))
                    score2 = int(float(score2))
                    average = (score1 + score2) / 2
                    if average >= 90:
                        grade = 'A'
                    elif average >= 80:
                        grade = 'B'
                    elif average >= 70:
                        grade = 'C'
                    elif average >= 60:
                        grade = 'D'
                    else:
                        grade = 'F'
                    slist.append({
                        'id': student_id,
                        'name': name,
                        'score1': score1,
                        'score2': score2,
                        'average': average,
                        'grade': grade
                    })
    except FileNotFoundError:
        print("")
    except Exception as e:
        print("")
    return slist

def main():
    while True:
        filename = input("불러올 파일명 입력: ").strip()
        if not filename:
            filename = 'students.txt'
            break
        if ' ' in filename:
            print("")
        else:
            break

    stu_list = load_students(filename)

    while True:
        command = input("# ").strip().lower()
        if command == "show":
            show(stu_list)
        elif command == "search":
            search(stu_list)
        elif command == "changescore":
            changeScore(stu_list)
        elif command == "searchgrade":
            searchGrade(stu_list)
        elif command == "add":
            add(stu_list)
        elif command == "remove":
            remove(stu_list)
        elif command == "quit":
            quit(stu_list)
            break
        else:
            print("")

if __name__ == "__main__":
    main()


# In[ ]:




